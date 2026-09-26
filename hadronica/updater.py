"""Update checks and in-place updates from the GitHub releases of bonquifo/Hadronica.

Once a day at most, a background thread asks GitHub's API for the latest release
and compares its tag (``v1.2`` or ``v1.2.0``) with ``hadronica.__version__``. If it
is newer, the status bar offers the update. The executable updates itself:

1. download the release's ``Hadronica.exe`` next to the running one, hashing it
   on the fly, and check it against the ``Hadronica.exe.sha256`` published with
   the same release (a mismatched, truncated, or non-Windows file is discarded
   and nothing changes);
2. rename the running executable to ``Hadronica.old.exe`` (Windows allows
   renaming a running program, not overwriting it) and move the new file into
   its place, putting the old one back if that fails;
3. start the new executable and quit. The next start deletes ``Hadronica.old.exe``.

Running from source, the same check only points to ``git pull`` and the release
page. Nothing here ever blocks the interface: a failed background check is
silent, and the app works offline exactly as before.

Settings and the last result live in %LOCALAPPDATA%\\Hadronica\\updater.json.
HADRONICA_NO_UPDATE_CHECK=1 turns the automatic check off. HADRONICA_UPDATE_URL
points the check at another release API (used by the tests, with a local server).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser

from hadronica import __version__

REPO = "bonquifo/Hadronica"
DEFAULT_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
EXE_NAME = "Hadronica.exe"
CHECKSUM_NAME = EXE_NAME + ".sha256"
CHECK_INTERVAL_S = 24 * 3600
TIMEOUT_S = 15
CHUNK = 256 * 1024
# Release downloads are served from these hosts (browser_download_url redirects to the second).
TRUSTED_HOSTS = ("github.com", "githubusercontent.com")


def api_url() -> str:
    return os.environ.get("HADRONICA_UPDATE_URL") or DEFAULT_API_URL


def parse_version(text: str) -> tuple[int, ...] | None:
    """``"v1.2"`` → (1, 2, 0). Pre-release or malformed tags give None."""
    match = re.fullmatch(r"v?(\d+(?:\.\d+){0,3})", (text or "").strip())
    if not match:
        return None
    parts = tuple(int(p) for p in match.group(1).split("."))
    return parts + (0,) * (3 - len(parts)) if len(parts) < 3 else parts


def is_newer(candidate: str, current: str) -> bool:
    new, now = parse_version(candidate), parse_version(current)
    return new is not None and now is not None and new > now


def display_version(tag: str) -> str:
    return tag[1:] if tag[:1] in "vV" else tag


def notes_excerpt(notes: str, max_lines: int = 8) -> list[str]:
    """Release notes as plain text lines: Markdown emphasis, code marks, and link targets removed."""
    lines = []
    for raw in (notes or "").replace("\r", "").split("\n"):
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
        text = re.sub(r"[*`_#>]+", "", text).strip()
        if text:
            lines.append(text)
        if len(lines) == max_lines:
            break
    return lines


def _trusted(url: str) -> bool:
    parts = urllib.parse.urlparse(url)
    override = os.environ.get("HADRONICA_UPDATE_URL")
    if override:
        # Testing against a local release server: only that server is trusted.
        return parts.netloc == urllib.parse.urlparse(override).netloc
    host = parts.hostname or ""
    return parts.scheme == "https" and any(host == h or host.endswith("." + h) for h in TRUSTED_HOSTS)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={
        "User-Agent": f"Hadronica/{__version__} (+https://github.com/{REPO})",
        "Accept": "application/vnd.github+json",
    })


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(_request(url), timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def open_stream(url: str):
    """A readable response with ``.read(n)`` and ``.headers``; the caller closes it."""
    return urllib.request.urlopen(_request(url), timeout=TIMEOUT_S)


class Release:
    def __init__(self, tag: str, page_url: str, notes: str = "", exe_url: str = "", exe_size: int = 0,
                 checksum_url: str = ""):
        self.tag = tag
        self.version = display_version(tag)
        self.page_url = page_url or RELEASES_PAGE
        self.notes = notes
        self.exe_url = exe_url
        self.exe_size = exe_size
        self.checksum_url = checksum_url

    @classmethod
    def from_api(cls, data: dict) -> "Release":
        assets = {a.get("name"): a for a in data.get("assets") or [] if isinstance(a, dict)}
        exe, checksum = assets.get(EXE_NAME) or {}, assets.get(CHECKSUM_NAME) or {}
        return cls(str(data.get("tag_name") or ""), str(data.get("html_url") or ""), str(data.get("body") or ""),
                   str(exe.get("browser_download_url") or ""), int(exe.get("size") or 0),
                   str(checksum.get("browser_download_url") or ""))

    def to_dict(self) -> dict:
        return {"tag": self.tag, "page_url": self.page_url, "notes": self.notes, "exe_url": self.exe_url,
                "exe_size": self.exe_size, "checksum_url": self.checksum_url}

    @classmethod
    def from_dict(cls, data: dict) -> "Release":
        return cls(data.get("tag", ""), data.get("page_url", ""), data.get("notes", ""), data.get("exe_url", ""),
                   int(data.get("exe_size") or 0), data.get("checksum_url", ""))

    @property
    def installable(self) -> bool:
        return bool(self.exe_url and self.checksum_url)


def default_state_file() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "Hadronica", "updater.json")


def frozen_exe() -> str | None:
    """The running Hadronica.exe, or None when running from source."""
    return sys.executable if getattr(sys, "frozen", False) else None


def old_path(exe_path: str) -> str:
    root, ext = os.path.splitext(exe_path)
    return f"{root}.old{ext}"


def download_path(exe_path: str) -> str:
    return exe_path + ".download"


def cleanup_after_update(exe_path: str | None = None) -> None:
    """Remove the previous executable and any interrupted download (run at every start)."""
    exe_path = exe_path or frozen_exe()
    if not exe_path:
        return
    for path in (old_path(exe_path), download_path(exe_path)):
        try:
            os.remove(path)
        except OSError:
            pass


def launch(exe_path: str, args=()) -> subprocess.Popen:
    """Start an executable as a new, independent program.

    A one-file PyInstaller program started from inside another would inherit its
    unpacked environment; PYINSTALLER_RESET_ENVIRONMENT makes it start clean.
    """
    env = dict(os.environ)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([exe_path, *args], env=env, cwd=os.path.dirname(exe_path) or None, close_fds=True,
                            creationflags=flags)


class Updater:
    """Update state for the interface. Network work runs on a thread; the UI only reads attributes.

    status: "idle" (nothing to show), "checking", "current", "available",
    "downloading", "ready" (verified download waiting to be installed), "error".
    """

    def __init__(self, *, current: str = __version__, exe_path: str | None = None, state_file: str | None = None,
                 fetch=fetch_json, opener=open_stream, now=time.time):
        self.current = current
        self.exe_path = exe_path if exe_path is not None else frozen_exe()
        self.state_file = state_file or default_state_file()
        self._fetch = fetch
        self._open = opener
        self._now = now
        self.status = "idle"
        self.release: Release | None = None
        self.message = ""
        self.progress = 0.0
        self.received = 0
        self.checked_at = 0.0
        self._thread: threading.Thread | None = None
        self.state = self._load_state()
        self.checked_at = float(self.state.get("last_check") or 0.0)

    # -- persisted settings ------------------------------------------------

    def _load_state(self) -> dict:
        try:
            with open(self.state_file, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=1)
        except OSError:
            pass

    @property
    def auto_check(self) -> bool:
        if os.environ.get("HADRONICA_NO_UPDATE_CHECK") == "1":
            return False
        return bool(self.state.get("auto_check", True))

    def set_auto_check(self, value: bool) -> None:
        self.state["auto_check"] = bool(value)
        self._save_state()

    @property
    def can_self_update(self) -> bool:
        return bool(self.exe_path) and self.release is not None and self.release.installable

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- checking --------------------------------------------------------------

    def start_background_check(self) -> None:
        """The automatic check at start: at most once a day, silent on failure."""
        if not self.auto_check or self.busy:
            return
        if self._now() - self.checked_at < CHECK_INTERVAL_S:
            cached = self.state.get("latest")
            if isinstance(cached, dict):
                self._offer(Release.from_dict(cached), manual=False)
            return
        self._run(self.check_now, False)

    def check_in_background(self) -> None:
        """The "Check now" button: always asks GitHub, and reports failures."""
        if not self.busy:
            self._run(self.check_now, True)

    def check_now(self, manual: bool = True) -> str:
        self.status, self.message = "checking", ""
        try:
            release = Release.from_api(self._fetch(api_url()))
        except Exception as exc:  # network, TLS, rate limit, bad JSON: the app carries on
            self.status = "error" if manual else "idle"
            self.message = f"Could not reach GitHub ({_reason(exc)})."
            return self.status
        self.checked_at = self._now()
        self.state["last_check"] = self.checked_at
        self.state["latest"] = release.to_dict()
        self._save_state()
        self._offer(release, manual=manual)
        return self.status

    def _offer(self, release: Release, *, manual: bool) -> None:
        self.release = release
        skipped = self.state.get("skipped") == release.tag
        if is_newer(release.tag, self.current) and (manual or not skipped):
            self.status = "available"
        else:
            self.status = "current" if manual or not is_newer(release.tag, self.current) else "idle"
        self.message = ""

    def skip(self) -> None:
        """Hide this release until a newer one appears (a manual check still shows it)."""
        if self.release is not None:
            self.state["skipped"] = self.release.tag
            self._save_state()
        self.status = "idle"

    # -- downloading and installing ------------------------------------------------

    def download_in_background(self) -> None:
        if self.can_self_update and not self.busy:
            self._run(self.download, None)

    def download(self, _unused=None) -> str:
        """Download and verify the new executable next to the running one; status becomes "ready"."""
        release, exe = self.release, self.exe_path
        if release is None or not exe or not release.installable:
            return self._fail("This release has no installable Windows executable.")
        if not (_trusted(release.exe_url) and _trusted(release.checksum_url)):
            return self._fail("The release points to an unexpected download address; nothing was downloaded.")
        self.status, self.progress, self.received = "downloading", 0.0, 0
        target = download_path(exe)
        try:
            expected = self._expected_sha256(release.checksum_url)
            digest = hashlib.sha256()
            with self._open(release.exe_url) as response, open(target, "wb") as out:
                total = release.exe_size or int(response.headers.get("Content-Length") or 0)
                first = b""
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    if not first:
                        first = chunk[:2]
                    out.write(chunk)
                    digest.update(chunk)
                    self.received += len(chunk)
                    self.progress = min(1.0, self.received / total) if total else 0.0
        except Exception as exc:
            _remove(target)
            return self._fail(f"The download failed ({_reason(exc)}); nothing was changed.")
        if release.exe_size and self.received != release.exe_size:
            _remove(target)
            return self._fail("The download was incomplete; nothing was changed.")
        if digest.hexdigest() != expected:
            _remove(target)
            return self._fail("The download did not match its published SHA-256 checksum; nothing was changed.")
        if first != b"MZ":
            _remove(target)
            return self._fail("The download is not a Windows program; nothing was changed.")
        self.progress, self.status = 1.0, "ready"
        return self.status

    def _expected_sha256(self, url: str) -> str:
        with self._open(url) as response:
            text = response.read(4096).decode("ascii", "replace")
        match = re.search(r"\b[0-9a-fA-F]{64}\b", text)
        if not match:
            raise ValueError("the checksum file has no SHA-256")
        return match.group(0).lower()

    def install(self) -> None:
        """Swap the verified download in for the running executable. Raises OSError, leaving the old one."""
        exe = self.exe_path
        new, old = download_path(exe), old_path(exe)
        _remove(old)
        os.replace(exe, old)
        try:
            os.replace(new, exe)
        except OSError:
            os.replace(old, exe)
            raise

    def install_and_restart(self, args=()) -> bool:
        """Install the verified download and start it. True means the caller should now quit."""
        try:
            self.install()
        except OSError as exc:
            self._fail(f"Could not replace {os.path.basename(self.exe_path)} ({_reason(exc)}). "
                       "Download the new version from the release page instead.")
            return False
        try:
            launch(self.exe_path, args)
        except OSError as exc:
            self._fail(f"Hadronica {self.release.version} is installed but did not start ({_reason(exc)}). "
                       "Start it again yourself.")
            return False
        return True

    def open_release_page(self) -> None:
        webbrowser.open(self.release.page_url if self.release else RELEASES_PAGE)

    # -- helpers -------------------------------------------------------------

    def _run(self, target, argument) -> None:
        self._thread = threading.Thread(target=target, args=(argument,), daemon=True, name="hadronica-updater")
        self._thread.start()

    def _fail(self, message: str) -> str:
        self.status, self.message = "error", message
        return self.status


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _reason(exc: BaseException) -> str:
    text = str(getattr(exc, "reason", "") or exc) or type(exc).__name__
    return text if len(text) < 90 else text[:87] + "…"


def run_cli_update(report_path: str, then_args=()) -> int:
    """``Hadronica.exe --update [report.json] [--then ARGS…]``: update without the window.

    Checks GitHub, installs a newer release if there is one, and writes a JSON
    report. With ``--then``, starts the new executable with ARGS afterwards
    (the same way the window restarts after an update). Exit code 0 means the
    program is up to date afterwards.
    """
    updater = Updater()
    report = {"current": updater.current, "executable": updater.exe_path}
    if not updater.exe_path:
        report.update(ok=False, error="Only Hadronica.exe updates itself; from source, use git pull.")
    else:
        status = updater.check_now(manual=True)
        report["latest"] = updater.release.tag if updater.release else None
        if status == "available":
            if updater.download() == "ready":
                try:
                    updater.install()
                    report.update(ok=True, updated=True, installed=updater.release.version)
                    if then_args:
                        launch(updater.exe_path, then_args)
                except OSError as exc:
                    report.update(ok=False, error=f"install failed: {_reason(exc)}")
            else:
                report.update(ok=False, error=updater.message)
        elif status == "current":
            report.update(ok=True, updated=False)
        else:
            report.update(ok=False, error=updater.message)
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    except OSError:
        pass
    return 0 if report.get("ok") else 1
