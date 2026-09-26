#!/usr/bin/env bash
# One-time MadGraph5_aMC@NLO setup: NLO loop libraries and the NLO PDF set.
set -euo pipefail
M=$HOME/micromamba/envs/hadronica-mg5
H=$HOME/micromamba/envs/hadronica-hep
export PATH=$M/bin:$PATH
for env in $M $H; do
    DATA="$($env/bin/lhapdf-config --datadir)"
    if [ ! -d "$DATA/NNPDF31_nlo_as_0118" ]; then
        curl -Ls "https://lhapdfsets.web.cern.ch/current/NNPDF31_nlo_as_0118.tar.gz" | tar -xz -C "$DATA"
    fi
    grep -h "NNPDF31_nlo_as_0118" "$DATA/pdfsets.index" | head -1
done
# MadGraph 3.5 uses Fortran constructs (e.g. the "$" edit descriptor) that
# gfortran 15 rejects at run time; pin the gcc/gfortran 13 series it is tested
# with. Loop libraries built by another compiler are rebuilt below.
if ! "$M/bin/gfortran" --version 2>/dev/null | grep -q " 13\." || [ ! -x "$M/bin/mg5_aMC" ]; then
    MAMBA_ROOT_PREFIX=$HOME/micromamba $HOME/.local/bin/micromamba install -y -q -n hadronica-mg5 -c conda-forge         mg5amcnlo "gfortran=13" "gcc=13" "gxx=13"
    rm -rf "$M/MG5_aMC/HEPTools/ninja" "$M/MG5_aMC/HEPTools/collier" "$M/MG5_aMC/HEPTools/oneloop"         "$M/MG5_aMC/HEPTools/lib"
fi
# The loop libraries must all be present (MadGraph looks for them in HEPTools/lib).
for lib in libavh_olo libninja libcollier; do
    if ! ls "$M/MG5_aMC/HEPTools/lib/$lib".* > /dev/null 2>&1; then
        rm -rf "$M/MG5_aMC/HEPTools/ninja" "$M/MG5_aMC/HEPTools/collier" "$M/MG5_aMC/HEPTools/oneloop"
        break
    fi
done
# MC@NLO subprocesses compile against FastJet and need its headers and
# fastjet-config, which the conda-forge runtime package does not ship.
# Build the official 3.5.1 release into the environment.
if [ ! -x "$M/bin/fastjet-config" ]; then
    FJ=$(mktemp -d)
    curl -Ls https://fastjet.fr/repo/fastjet-3.5.1.tar.gz | tar -xz -C "$FJ"
    (cd "$FJ/fastjet-3.5.1" && CC=$M/bin/x86_64-conda-linux-gnu-cc CXX=$M/bin/x86_64-conda-linux-gnu-c++         ./configure --prefix="$M" --disable-auto-ptr --enable-allcxxplugins > "$FJ/configure.log" 2>&1 &&         make -j"$(nproc)" > "$FJ/make.log" 2>&1 && make install > "$FJ/install.log" 2>&1) ||         { echo "FastJet build failed; see $FJ"; exit 1; }
    rm -rf "$FJ"
fi
# conda-forge builds IREGI against an external OneLOop (upstream bundles its own
# copy), so NLO executables must also link libavh_olo after -liregi.
OPTS="$M/MG5_aMC/Template/NLO/Source/make_opts.inc"
if ! grep -q "lavh_olo" "$OPTS"; then
    sed -i "s|libcuttools=-lcts %(link_tir_libs)s|libcuttools=-lcts %(link_tir_libs)s -L$M/MG5_aMC/HEPTools/lib -lavh_olo|" "$OPTS"
fi
grep -q "lavh_olo" "$OPTS" || { echo "could not patch $OPTS"; exit 1; }
# Newer libgfortran runtimes reject the non-standard '$' edit descriptor that a
# few MadGraph templates still use; rewrite those WRITEs with advance='no'.
DOLLAR_FILES=$(grep -rlE '\$[[:space:]]*\)' --include=*.f --include=*.f90 --include=*.inc "$M/MG5_aMC/Template" "$M/MG5_aMC/madgraph" || true)
"$M/bin/python" "$(dirname "$0")/fix_dollar_formats.py" $DOLLAR_FILES > /dev/null
cd /tmp
cat > /tmp/mg5_install.txt <<CMD
set fastjet $M/bin/fastjet-config
set auto_update 0
set automatic_html_opening False
set nb_core 30
set run_mode 2
set lhapdf_py3 $M/bin/lhapdf-config
set fortran_compiler $M/bin/gfortran
install oneloop
install ninja
install collier
CMD
mg5_aMC /tmp/mg5_install.txt 2>&1 | tail -5
