#!/usr/bin/env bash
# Generate an MC@NLO event sample with MadGraph5_aMC@NLO for SMLab.
#
#   generate_nlo.sh <process: ttbar|dy|w> <sqrt_s in GeV> <events>
#
# Output: ~/smlab-cache/nlo/<process>_<sqrt_s>/events.lhe plus info.json with
# the NLO cross section, its statistical error, and the scale uncertainty.
# Inputs follow PDG 2026: m_t = 172.60, M_Z = 91.1879, widths as listed, and
# α⁻¹ = 132.04 chosen so the G_F scheme reproduces M_W = 80.3625 GeV at tree
# level. PDFs: NNPDF3.1 NLO (LHAID 303400). Shower matching: PYTHIA8.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROC="$1"
SQRT_S="$2"
NEVENTS="$3"
M=$HOME/micromamba/envs/smlab-mg5
export PATH=$M/bin:$PATH
OUT=$HOME/smlab-cache/nlo/${PROC}_${SQRT_S}
mkdir -p "$(dirname "$OUT")"
EBEAM=$(python3 -c "print($SQRT_S / 2)")

MADSPIN=OFF
META="{}"
case "$PROC" in
    ttbar) GENERATE="generate p p > t t~ [QCD]"; EXTRA="" ;;
    ttbar_ms)
        # Top decays by MadSpin (decay_madspin.sh, after generation): full spin
        # correlations and off-shell Breit-Wigner tails.
        GENERATE="generate p p > t t~ [QCD]"; EXTRA=""; MADSPIN=ON
        META='{"madspin": true}' ;;
    dy)    GENERATE="generate p p > l+ l- [QCD]"; EXTRA="set mll_sf 60" ;;
    dy_fxfx)
        # FxFx: Z + 0, 1, 2 jets, each at NLO, merged at Qcut = 20 GeV (ptj = 10 GeV).
        GENERATE="generate p p > l+ l- [QCD] @0
add process p p > l+ l- j [QCD] @1
add process p p > l+ l- j j [QCD] @2"
        EXTRA="set mll_sf 60
set ickkw 3
set ptj 10
set jetalgo 1
set jetradius 1.0
set maxjetflavor 5"
        META='{"fxfx": {"qcut": 20.0, "ptj": 10.0, "njmax": 2}}' ;;
    w)     GENERATE="generate p p > l+ vl [QCD]
add process p p > l- vl~ [QCD]"; EXTRA="" ;;
    *) echo "unknown process $PROC"; exit 1 ;;
esac

WORK=$(mktemp -d)
cat > "$WORK/cmd.txt" <<CMD
set auto_update 0
set automatic_html_opening False
set nb_core 30
set run_mode 2
import model loop_sm-no_b_mass
define p = p b b~
define j = p
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm
define vl~ = ve~ vm~
$GENERATE
output $OUT.mg5 -f
launch
fixed_order=OFF
shower=OFF
madspin=OFF
reweight=OFF
done
set nevents $NEVENTS
set ebeam1 $EBEAM
set ebeam2 $EBEAM
set pdlabel lhapdf
set lhaid 303400
set parton_shower PYTHIA8
set iseed 20260923
set mt 172.6
set ymt 172.6
set wt 1.42
set mz 91.1879
set wz 2.4955
set ww 2.14
set aewm1 132.04
set gf 1.1663785e-5
$EXTRA
done
CMD
START=$(date +%s)
# MadGraph writes history and debug files into the current directory.
cd "$WORK"
mg5_aMC "$WORK/cmd.txt" > "$OUT.log" 2>&1
cd /
rm -rf "$WORK"
if [ "$MADSPIN" = ON ]; then
    # Standalone MadSpin on the undecayed events; writes $OUT/events.lhe and info.json.
    exec bash "$HERE/decay_madspin.sh" "${PROC}_${SQRT_S}" "$(( $(date +%s) - START ))"
fi
RUN="$OUT.mg5/Events/run_01"
mkdir -p "$OUT"
gunzip -c "$RUN/events.lhe.gz" > "$OUT/events.lhe"
"$M/bin/python" "$HERE/sample_info.py" "$OUT" "$PROC" "$SQRT_S" "$(( $(date +%s) - START ))" "$OUT.log" "$META"
