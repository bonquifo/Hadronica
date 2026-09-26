"""Publish a Hadronica release that running copies can update to.

    python tools/release.py --notes-file notes.md [--dry-run]

The tag is ``v`` + hadronica.__version__, so bump that first. The script refuses
to release uncommitted or unpushed work or an existing tag, then:

1. builds Hadronica.exe with build_exe.ps1 into dist/release/;
2. runs the executable's own --selftest and stops unless every check passes;
3. writes Hadronica.exe.sha256, which the in-app updater requires before it
   installs anything;
4. creates the GitHub release with both files (gh CLI).

--dry-run does steps 1–3 and prints what it would publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hadronica import __version__  # noqa: E402
from hadronica.updater import CHECKSUM_NAME, EXE_NAME, REPO, parse_version  # noqa: E402

OUT = os.path.join(ROOT, "dist", "release")


def run(cmd, **kw) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, **kw).stdout.strip()


def fail(message: str) -> None:
    sys.exit(f"release: {message}")


def preflight(tag: str) -> None:
    if parse_version(tag) is None:
        fail(f"{__version__!r} is not a plain version number")
    if run(["git", "status", "--porcelain"]):
        fail("the working tree has uncommitted changes")
    run(["git", "fetch", "-q", "origin"])
    if run(["git", "rev-parse", "HEAD"]) != run(["git", "rev-parse", "origin/main"]):
        fail("HEAD is not what origin/main has; push (or pull) first")
    tags = json.loads(run(["gh", "release", "list", "--repo", REPO, "--json", "tagName", "--limit", "200"]))
    if any(t["tagName"] == tag for t in tags):
        fail(f"release {tag} already exists; bump hadronica.__version__")
    latest = max((parse_version(t["tagName"]) or (0,) for t in tags), default=(0,))
    if parse_version(tag) <= latest:
        fail(f"{tag} is not newer than the latest release; running copies would not update to it")


def build() -> str:
    env = dict(os.environ, HADRONICA_DIST=OUT)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    os.path.join(ROOT, "build_exe.ps1")], cwd=ROOT, env=env, check=True)
    exe = os.path.join(OUT, EXE_NAME)
    if not os.path.isfile(exe):
        fail("the build produced no executable")
    return exe


def selftest(exe: str) -> None:
    report = os.path.join(OUT, "selftest.json")
    code = subprocess.run([exe, "--selftest", report], cwd=OUT).returncode
    with open(report, encoding="utf-8") as handle:
        result = json.load(handle)
    failed = [c["name"] for c in result.get("checks", []) if not c.get("ok")]
    if code != 0 or not result.get("passed") or failed:
        fail(f"the executable's self-test failed: {failed or code}")
    print(f"self-test passed ({len(result['checks'])} checks)")


def checksum(exe: str) -> str:
    digest = hashlib.sha256()
    with open(exe, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    path = os.path.join(OUT, CHECKSUM_NAME)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest.hexdigest()}  {EXE_NAME}\n")
    print(f"sha256 {digest.hexdigest()}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notes-file", required=True, help="Markdown release notes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tag = f"v{__version__}"
    if not args.dry_run:
        preflight(tag)
    exe = build()
    selftest(exe)
    sums = checksum(exe)
    if args.dry_run:
        print(f"dry run: would publish {tag} with {exe} and {sums}")
        return
    url = run(["gh", "release", "create", tag, exe, sums, "--repo", REPO, "--target", "main",
               "--title", f"Hadronica {__version__}", "--notes-file", args.notes_file])
    print(url)


if __name__ == "__main__":
    main()
