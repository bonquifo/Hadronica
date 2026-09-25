# Build a windowed SMLab.exe next to this script. The WSL-side scripts
# (PYTHIA worker, Delphes bridge, Rivet validation, MadGraph tools) and the
# baseline validation results and shower tune are bundled so the executable is self-contained.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$data = @(
    "hep\worker.py;hep", "hep\detector.py;hep", "hep\validate.py;hep", "hep\setup_wsl.sh;hep",
    "hep\check_env.py;hep", "hep\validation\results.json;hep\validation",
    "hep\mg5\generate_nlo.sh;hep\mg5", "hep\mg5\produce_all.sh;hep\mg5", "hep\mg5\setup_mg5.sh;hep\mg5",
    "hep\mg5\sample_info.py;hep\mg5", "hep\mg5\fix_dollar_formats.py;hep\mg5",
    "hep\mg5\decay_madspin.sh;hep\mg5", "hep\mg5\restore_weights.py;hep\mg5", "hep\tune.py;hep",
    "hep\ext\smlab_fxfx.cpp;hep\ext", "hep\ext\build_fxfx.sh;hep\ext",
    "smlab\assets\window_icon.png;smlab\assets"
)
# The SMLab shower tune (written by hep/tune.py), applied by the worker in NLO mode.
if (Test-Path "hep\validation\tune.json") { $data += "hep\validation\tune.json;hep\validation" }
# The exe's own icon (Explorer, shortcuts, pinned taskbar), all sizes 16-256 px; see tools/make_icon.py.
$pyiArgs = @("--noconfirm", "--clean", "--windowed", "--onefile", "--name", "SMLab", "--collect-all", "pygame",
             "--icon", "smlab\assets\icon.ico")
if ($env:SMLAB_DIST) { $pyiArgs += @("--distpath", $env:SMLAB_DIST) }
foreach ($item in $data) { $pyiArgs += @("--add-data", $item) }
# PyInstaller logs to stderr, which Windows PowerShell 5.1 turns into terminating
# errors under "Stop"; judge success by the exit code instead.
$ErrorActionPreference = "Continue"
python -m PyInstaller @pyiArgs run_smlab.py 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
Write-Host "Built SMLab.exe"
