"""Run the WSL-side test suite (tests/hep) from Windows, so one pytest run covers both halves.

tests/hep drives the real PYTHIA 8, pyhepmc, gfortran, and the MadGraph5_aMC@NLO
samples, which exist only inside WSL. Skipped when WSL or the environment is missing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hadronica import engine
from tests.test_engine import _wsl_engine_available


@pytest.mark.skipif(not _wsl_engine_available(), reason="PYTHIA environment in WSL is not installed")
def test_wsl_suite_passes():
    root = engine.worker_path_in_wsl().rsplit("/hep/", 1)[0]
    command = f"cd '{root}' && {engine.ENV_PYTHON} -m pytest -q -p no:cacheprovider -rs tests/hep"
    done = subprocess.run(["wsl.exe", "-d", engine.WSL_DISTRO, "--", "bash", "-lc", command],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    sys.stdout.write(done.stdout[-6000:])
    assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-2000:]
    assert " passed" in done.stdout and " failed" not in done.stdout
