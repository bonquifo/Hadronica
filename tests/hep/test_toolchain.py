"""The MadGraph toolchain fixes, checked by compiling and running real Fortran, and shell-script syntax."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import pytest

from hep_paths import MG5_ENV, ROOT

GFORTRAN = os.path.join(MG5_ENV, "bin", "gfortran")
FIXER = os.path.join(ROOT, "hep", "mg5", "fix_dollar_formats.py")

PROGRAM = """\
      program t
      integer n
      character*20 formstr
      n = 42
      write(*,'(a$)') 'ab'
      write(*,'(i5,1x,$)') n
      write(formstr,'(a,i1,a)') '(I',2,'$)'
      write(*,formstr) n
      write(*,*) 'end'
      end
"""


def _compile_and_run(tmp_path, source: str) -> subprocess.CompletedProcess:
    src = tmp_path / "t.f"
    src.write_text(source)
    subprocess.run([GFORTRAN, str(src), "-o", str(tmp_path / "t")], check=True, capture_output=True)
    return subprocess.run([str(tmp_path / "t")], capture_output=True, text=True)


@pytest.mark.skipif(not os.path.exists(GFORTRAN), reason="MadGraph environment not installed")
def test_madgraph_runtime_rejects_dollar_formats_and_the_fix_restores_the_output(tmp_path):
    original = _compile_and_run(tmp_path, PROGRAM)
    # The reason for the fixer: MadGraph's libgfortran aborts on the non-standard '$' descriptor.
    assert original.returncode != 0 and "Missing comma" in original.stderr
    fixed_source = tmp_path / "fixed.f"
    fixed_source.write_text(PROGRAM)
    subprocess.run([sys.executable, FIXER, str(fixed_source)], check=True, capture_output=True)
    fixed = _compile_and_run(tmp_path, fixed_source.read_text())
    assert fixed.returncode == 0, fixed.stderr
    # Non-advancing: everything on one line, exactly as the '$' formats intended.
    assert fixed.stdout.splitlines() == ["ab   42 42 end"]


@pytest.mark.skipif(not os.path.isdir(os.path.join(MG5_ENV, "MG5_aMC")), reason="MadGraph not installed")
def test_no_unfixed_dollar_format_is_left_in_the_madgraph_templates():
    roots = [os.path.join(MG5_ENV, "MG5_aMC", "Template"), os.path.join(MG5_ENV, "MG5_aMC", "madgraph")]
    files = [f for root in roots for pattern in ("*.f", "*.inc", "*.f90")
             for f in glob.glob(os.path.join(root, "**", pattern), recursive=True) if os.path.isfile(f)]
    report = subprocess.run([sys.executable, FIXER, "--dry-run", *files], capture_output=True, text=True, check=True)
    assert " 0 replaced" in report.stdout and all(" 0 replaced, 0 left" in line for line in report.stdout.splitlines())


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", sorted(glob.glob(os.path.join(ROOT, "hep", "**", "*.sh"), recursive=True)),
                         ids=lambda path: os.path.relpath(path, ROOT))
def test_shell_scripts_parse(script):
    checked = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert b"\r\n" not in open(script, "rb").read()  # CRLF line endings break bash in WSL
