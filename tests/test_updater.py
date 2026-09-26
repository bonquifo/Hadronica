"""The updater: version comparison, daily checks, verified downloads, and the executable swap."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from hadronica import __version__, updater
from hadronica.updater import Release, Updater, is_newer, notes_excerpt, parse_version

EXE = b"MZ" + b"\x90" * 5000  # a stand-in Windows executable
SHA = hashlib.sha256(EXE).hexdigest()
API = "https://api.github.com/repos/bonquifo/Hadronica/releases/latest"
EXE_URL = "https://github.com/bonquifo/Hadronica/releases/download/v9.0/Hadronica.exe"
SUM_URL = EXE_URL + ".sha256"


def api_release(tag="v9.0", *, exe=True, checksum=True, size=len(EXE)):
    assets = []
    if exe:
        assets.append({"name": "Hadronica.exe", "browser_download_url": EXE_URL, "size": size})
    if checksum:
        assets.append({"name": "Hadronica.exe.sha256", "browser_download_url": SUM_URL, "size": 81})
    return {"tag_name": tag, "html_url": "https://github.com/bonquifo/Hadronica/releases/tag/" + tag,
            "body": "**New:** faster [histograms](https://example.org)\n\n- `B` runs 200", "assets": assets}


class Stream(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def opener(files: dict):
    def open_url(url):
        return Stream(files[url])
    return open_url


def make(tmp_path, data=None, *, files=None, exe=True, now=1_000_000.0, current="1.2.0"):
    calls = []

    def fetch(url):
        calls.append(url)
        if isinstance(data, Exception):
            raise data
        return data if data is not None else api_release()

    exe_path = None
    if exe:
        exe_path = tmp_path / "Hadronica.exe"
        exe_path.write_bytes(b"MZ old build")
        exe_path = str(exe_path)
    u = Updater(current=current, exe_path=exe_path, state_file=str(tmp_path / "state" / "updater.json"),
                fetch=fetch, opener=opener(files or {EXE_URL: EXE, SUM_URL: f"{SHA}  Hadronica.exe\n".encode()}),
                now=lambda: now)
    return u, calls


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("HADRONICA_UPDATE_URL", raising=False)
    monkeypatch.delenv("HADRONICA_NO_UPDATE_CHECK", raising=False)


def test_versions_compare_numerically_and_tags_match_the_app_version():
    assert parse_version("v1.1") == (1, 1, 0)
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("v1.2-beta") is None and parse_version("") is None
    assert is_newer("v1.10", "1.9.0") and is_newer("v1.2", "1.1.9")
    assert not is_newer("v1.2", "1.2.0") and not is_newer("v1.1", "1.2.0") and not is_newer("junk", "1.0")
    # The in-app version must be comparable with the release tags (it once read 2.0.0 while releases were v1.x).
    assert parse_version(__version__) is not None and parse_version(__version__) > parse_version("v1.1")


def test_a_newer_release_is_offered_and_remembered(tmp_path):
    u, calls = make(tmp_path)
    assert u.check_now(manual=False) == "available"
    assert calls == [API]
    assert u.release.version == "9.0" and u.can_self_update
    saved = json.loads((tmp_path / "state" / "updater.json").read_text())
    assert saved["last_check"] == 1_000_000.0 and saved["latest"]["tag"] == "v9.0"


def test_the_latest_release_is_current(tmp_path):
    u, _ = make(tmp_path, api_release("v1.2"))
    assert u.check_now() == "current"


def test_the_automatic_check_runs_at_most_once_a_day(tmp_path):
    u, calls = make(tmp_path)
    u.check_now(manual=False)
    later, calls2 = make(tmp_path, now=1_000_000.0 + 3600)
    later.start_background_check()
    assert calls2 == [] and later.status == "available"  # the cached answer, no request
    next_day, calls3 = make(tmp_path, now=1_000_000.0 + 25 * 3600)
    next_day.start_background_check()
    next_day._thread.join(5)
    assert calls3 == [API] and next_day.status == "available"


def test_the_automatic_check_can_be_turned_off(tmp_path, monkeypatch):
    u, calls = make(tmp_path)
    u.set_auto_check(False)
    u.start_background_check()
    assert calls == [] and u.status == "idle"
    assert json.loads((tmp_path / "state" / "updater.json").read_text())["auto_check"] is False
    monkeypatch.setenv("HADRONICA_NO_UPDATE_CHECK", "1")
    (tmp_path / "other").mkdir()
    v, calls = make(tmp_path / "other")
    v.start_background_check()
    assert calls == [] and not v.auto_check


def test_offline_is_silent_in_the_background_and_reported_on_request(tmp_path):
    u, _ = make(tmp_path, OSError("no route to host"))
    assert u.check_now(manual=False) == "idle" and u.release is None
    assert u.check_now(manual=True) == "error" and "Could not reach GitHub" in u.message


def test_a_skipped_release_stays_hidden_until_checked_by_hand(tmp_path):
    u, _ = make(tmp_path)
    u.check_now(manual=False)
    u.skip()
    again, _ = make(tmp_path, now=1_000_000.0 + 2 * 86400)
    assert again.check_now(manual=False) == "idle"
    assert again.check_now(manual=True) == "available"
    newer, _ = make(tmp_path, api_release("v9.1"), now=1_000_000.0 + 3 * 86400)
    assert newer.check_now(manual=False) == "available"


def test_from_source_there_is_nothing_to_install(tmp_path):
    u, _ = make(tmp_path, exe=False)
    assert u.check_now() == "available" and not u.can_self_update


def test_a_release_without_a_checksum_is_not_installable(tmp_path):
    u, _ = make(tmp_path, api_release(checksum=False))
    u.check_now()
    assert not u.can_self_update
    assert u.download() == "error" and "no installable" in u.message


def test_a_verified_download_replaces_the_executable_and_the_old_one_is_cleaned_up(tmp_path):
    u, _ = make(tmp_path)
    u.check_now()
    assert u.download() == "ready" and u.progress == 1.0
    assert (tmp_path / "Hadronica.exe.download").read_bytes() == EXE
    u.install()
    assert (tmp_path / "Hadronica.exe").read_bytes() == EXE
    assert (tmp_path / "Hadronica.old.exe").read_bytes() == b"MZ old build"
    assert not (tmp_path / "Hadronica.exe.download").exists()
    updater.cleanup_after_update(str(tmp_path / "Hadronica.exe"))
    assert not (tmp_path / "Hadronica.old.exe").exists()


@pytest.mark.parametrize("files, reason", [
    ({EXE_URL: EXE[:-10] + b"x" * 10, SUM_URL: SHA.encode()}, "checksum"),
    ({EXE_URL: EXE[:100], SUM_URL: SHA.encode()}, "incomplete"),
    ({EXE_URL: EXE, SUM_URL: b"not a hash"}, "download failed"),
    ({EXE_URL: b"PK" + EXE[2:], SUM_URL: hashlib.sha256(b"PK" + EXE[2:]).hexdigest().encode()}, "not a Windows program"),
])
def test_a_bad_download_changes_nothing(tmp_path, files, reason):
    u, _ = make(tmp_path, files=files)
    u.check_now()
    assert u.download() == "error" and reason in u.message
    assert (tmp_path / "Hadronica.exe").read_bytes() == b"MZ old build"
    assert not (tmp_path / "Hadronica.exe.download").exists()


def test_downloads_only_come_from_github(tmp_path):
    data = api_release()
    data["assets"][0]["browser_download_url"] = "https://evil.example.com/Hadronica.exe"
    u, _ = make(tmp_path, data)
    u.check_now()
    assert u.download() == "error" and "unexpected download address" in u.message
    assert updater._trusted("https://objects.githubusercontent.com/x") and not updater._trusted("http://github.com/x")


def test_a_failed_swap_puts_the_old_executable_back(tmp_path, monkeypatch):
    u, _ = make(tmp_path)
    u.check_now()
    u.download()
    real = updater.os.replace

    def replace(src, dst):
        if src.endswith(".download"):
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(updater.os, "replace", replace)
    assert u.install_and_restart() is False
    assert u.status == "error" and "release page" in u.message
    assert (tmp_path / "Hadronica.exe").read_bytes() == b"MZ old build"


def test_install_and_restart_starts_the_new_program_with_a_clean_environment(tmp_path, monkeypatch):
    u, _ = make(tmp_path)
    u.check_now()
    u.download()
    started = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    assert u.install_and_restart() is True
    (cmd, kw), = started
    assert cmd == [str(tmp_path / "Hadronica.exe")]
    assert kw["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"


def test_release_notes_are_shown_as_plain_text():
    assert notes_excerpt("**New:** faster [histograms](https://x)\n\n- `B` runs 200") == \
        ["New: faster histograms", "- B runs 200"]
    assert Release.from_api(api_release()).installable


# -- the interface ------------------------------------------------------------


def _app_with(u):
    from hadronica.app import LabApp

    app = LabApp(size=(1180, 800), headless=True, seed=3)
    app.updater = u
    app.draw()
    return app


def _click(app, action):
    rect = next(rect for rect, a, _p in reversed(app.hot) if a == action)
    app._click(rect.center)
    app.draw()


def test_the_status_bar_offers_the_update_and_the_dialog_installs_it(tmp_path, monkeypatch):
    import pygame

    u, _ = make(tmp_path)
    u.check_now(manual=False)
    app = _app_with(u)
    assert any(a == "update" for _r, a, _p in app.hot)  # the pill and the version
    _click(app, "update")
    assert app.show_update
    started = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    pygame.event.clear()
    _click(app, "upd-install")
    u._thread.join(5)
    assert u.status == "ready"
    app._poll_update()
    assert started == [[str(tmp_path / "Hadronica.exe")]]
    assert (tmp_path / "Hadronica.exe").read_bytes() == EXE
    assert any(e.type == pygame.QUIT for e in pygame.event.get())


def test_the_dialog_skips_closes_and_remembers_the_setting(tmp_path):
    import pygame

    u, _ = make(tmp_path)
    u.check_now(manual=False)
    app = _app_with(u)
    _click(app, "update")
    _click(app, "upd-auto")
    assert u.auto_check is False
    app._click((2, 2))  # outside the dialog
    assert not app.show_update
    _click(app, "update")
    app._key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode=""))
    assert not app.show_update
    _click(app, "update")
    _click(app, "upd-skip")
    assert not app.show_update and u.status == "idle"
    assert not any(a == "update" and r.w > 150 for r, a, _p in app.hot)  # the pill is gone


def test_no_update_interface_without_an_updater():
    from hadronica.app import LabApp

    app = LabApp(size=(1180, 800), headless=True, seed=3)
    app.draw()
    assert not any(a == "update" for _r, a, _p in app.hot)
