#!/usr/bin/env bash
# One-time setup of the research-grade engines for Hadronica, inside WSL (Ubuntu).
# Installs micromamba in ~/.local/bin and two conda-forge environments:
#   smlab-hep  PYTHIA 8.3, LHAPDF 6, HepMC3 (+ pyhepmc), FastJet, Delphes 3.5
#              with ROOT, uproot, Rivet 4.1 and YODA 2
#   smlab-mg5  MadGraph5_aMC@NLO 3.5 with gcc/gfortran 13, FastJet, and the
#              NLO loop libraries (see mg5/setup_mg5.sh)
# No sudo is needed.
#
#   wsl -d Ubuntu -- bash /mnt/c/ParticleCollision/hep/setup_wsl.sh
set -euo pipefail

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
MM="$HOME/.local/bin/micromamba"
if [ ! -x "$MM" ]; then
    mkdir -p "$HOME/.local"
    (cd "$HOME/.local" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba)
fi
if [ ! -x "$MAMBA_ROOT_PREFIX/envs/smlab-hep/bin/python" ]; then
    "$MM" create -y -n smlab-hep -c conda-forge python=3.12 pythia8 lhapdf hepmc3 fastjet delphes uproot numpy         rivet yoda pyhepmc
fi
if [ ! -x "$MAMBA_ROOT_PREFIX/envs/smlab-hep/bin/rivet" ]; then
    "$MM" install -y -n smlab-hep -c conda-forge rivet yoda pyhepmc
fi
# pytest runs the WSL-side test suite (tests/hep), driven by tests/test_wsl_suite.py.
"$MAMBA_ROOT_PREFIX/envs/smlab-hep/bin/python" -m pip install -q pytest
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/smlab-mg5" ]; then
    "$MM" create -y -n smlab-mg5 -c conda-forge python=3.11 mg5amcnlo lhapdf "gfortran=13" "gcc=13" "gxx=13"         make cmake six
fi
bash "$(dirname "$0")/mg5/setup_mg5.sh"

# LHAPDF sets are optional (PYTHIA's Monash tune uses its built-in NNPDF2.3 LO).
# Fetch one modern NNLO set for users who want to switch.
ENV="$MAMBA_ROOT_PREFIX/envs/smlab-hep"
DATA="$("$ENV/bin/lhapdf-config" --datadir)"
if [ ! -d "$DATA/NNPDF31_nnlo_as_0118" ]; then
    curl -Ls "https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118.tar.gz" | tar -xz -C "$DATA" || \
        echo "warning: could not download the NNPDF3.1 LHAPDF set (optional)"
fi
# The conda-forge Delphes package omits the include files that the CLICdet and
# muon-collider cards source. Take them from the matching upstream release.
CARDS="$ENV/cards"
if [ ! -d "$CARDS/CLIC" ] || [ ! -d "$CARDS/MuonCollider" ]; then
    TMP="$(mktemp -d)"
    curl -Ls "https://github.com/delphes/delphes/archive/refs/tags/3.5.1.tar.gz" | tar -xz -C "$TMP"
    cp -r "$TMP"/delphes-3.5.1/cards/CLIC "$TMP"/delphes-3.5.1/cards/MuonCollider "$CARDS"/
    rm -rf "$TMP"
fi
# PYTHIA's FxFx jet-matching hook for Python (the bindings do not expose it).
bash "$(dirname "$0")/ext/build_fxfx.sh"
"$ENV/bin/python" "$(dirname "$0")/check_env.py"
