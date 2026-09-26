#!/usr/bin/env bash
# Build the hadronica_fxfx Python module (hep/ext/hadronica_fxfx.cpp) into the hadronica-hep env.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
E=$HOME/micromamba/envs/hadronica-hep
# pybind11 internals must match pythia8.so (v12: pybind11 3.x).
"$E/bin/python" -m pip install -q "pybind11>=3,<4"
SITE=$("$E/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
SUFFIX=$("$E/bin/python" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
"$E/bin/x86_64-conda-linux-gnu-g++" -O2 -shared -fPIC -std=c++17 \
    $("$E/bin/python" -m pybind11 --includes) -I"$E/include" \
    "$HERE/hadronica_fxfx.cpp" -L"$E/lib" -lpythia8 -Wl,-rpath,"$E/lib" \
    -o "$SITE/hadronica_fxfx$SUFFIX"
"$E/bin/python" -c "import os, sys, pythia8, hadronica_fxfx; p = pythia8.Pythia(os.path.join(sys.prefix, 'share', 'Pythia8', 'xmldoc'), False); hadronica_fxfx.attach(p); print('hadronica_fxfx ok')"
