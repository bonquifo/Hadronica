"""SMLab's built-in Born engine against MadGraph5_aMC@NLO at leading order.

    ~/micromamba/envs/smlab-mg5/bin/python hep/compare/builtin_vs_madgraph.py [--events N]

MadGraph5_aMC@NLO (J. Alwall et al., JHEP 07 (2014) 079, arXiv:1405.0301; tree
level, G_F scheme: α = 1/132.04, on-shell sin²θ_W = 1 − M_W²/M_Z², fixed-width propagators, no QCD factor, no ISR) is the reference. SMLab's
engine is run twice:

* scheme-matched: its couplings switched to exactly MadGraph's (α, sin²θ_W, no
  QCD factor), so any remaining difference is an error in SMLab's formulas;
* as shipped: the improved Born approximation (running α(s), sin²θ_eff, QCD
  factor for quarks), whose differences from MadGraph are the scheme choice.

The pure QED processes (Bhabha without Z exchange, e⁺e⁻ → γγ, both with the 10°
fiducial cut, |η| < 2.4362) use α(0) in SMLab and are compared with MadGraph at
α = 1/137.036. The forward–backward asymmetry of e⁺e⁻ → μ⁺μ⁻ is compared on
MadGraph's unweighted events. Writes hep/compare/builtin_vs_madgraph.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from smlab import constants as C  # noqa: E402
from smlab import electroweak as EW  # noqa: E402
from smlab.generator import sigma_pb  # noqa: E402
from smlab.processes import BEAMS, process_by_id  # noqa: E402

MG5 = os.path.expanduser("~/micromamba/envs/smlab-mg5/bin/mg5_aMC")
ALPHA_GF = 1.0 / 132.04
M_W_GF = 80.3617  # derived by MadGraph from M_Z, G_F, α (see AUDIT_2026.md)
SIN2_OS = 1.0 - (M_W_GF / C.M_Z) ** 2
ETA_10DEG = -math.log(math.tan(math.radians(5.0)))  # θ = 10° ↔ |η| = 2.4362 for massless 2 → 2

# (SMLab process id, MadGraph process, √s values, scheme: "ew" or "qed")
CASES = [
    ("ff13", "e+ e- > mu+ mu-", (20.0, C.M_Z, 250.0, 500.0), "ew"),
    ("ff15", "e+ e- > ta+ ta-", (C.M_Z, 250.0), "ew"),
    ("ff2", "e+ e- > u u~", (C.M_Z, 250.0), "ew"),
    ("ff1", "e+ e- > d d~", (C.M_Z, 250.0), "ew"),
    ("ff5", "e+ e- > b b~", (C.M_Z, 250.0), "ew"),
    ("ff6", "e+ e- > t t~", (365.0, 500.0), "ew"),
    ("ff14", "e+ e- > vm vm~", (C.M_Z, 250.0), "ew"),
    ("ff12", "e+ e- > ve ve~", (C.M_Z, 250.0), "ew"),
    ("zh", "e+ e- > z h", (240.0, 365.0, 500.0), "ew"),
    ("bhabha", "e+ e- > e+ e- / z h", (C.M_Z, 250.0), "qed"),
    ("diphoton", "e+ e- > a a", (C.M_Z, 250.0), "qed"),
]

NO_CUTS = """set lpp1 0
set lpp2 0
set ptl 0
set pta 0
set ptj 0
set ptb 0
set etal -1
set etaa -1
set etaj -1
set etab -1
set drll 0
set draa 0
set drjj 0
set drbb 0
set dral 0
set drjl 0
set mmll 0
set mmjj 0
set mmbb 0
set use_syst False
set dynamical_scale_choice 3
"""


def params(scheme: str) -> str:
    aewm1 = 137.035999 if scheme == "qed" else 132.04
    return f"""set mz {C.M_Z}
set wz {C.GAMMA_Z}
set mt {C.M_T}
set wt {C.GAMMA_T}
set mh {C.M_H}
set wh {C.GAMMA_H}
set mta {C.M_TAU}
set mb {C.M_B}
set ymb {C.M_B}
set aewm1 {aewm1}
set gf {C.G_F}
"""


def run_madgraph(pid: str, process: str, energies, scheme: str, events: int, work: str) -> dict:
    out = os.path.join(work, pid)
    lines = ["set auto_update 0", "set automatic_html_opening False", "set nb_core 4",
             "import model sm", f"generate {process}", f"output {out} -f"]
    for energy in energies:
        lines += ["launch", "done", NO_CUTS.strip(), params(scheme).strip(),
                  f"set ebeam1 {energy / 2}", f"set ebeam2 {energy / 2}", f"set nevents {events}", "set iseed 11"]
        if scheme == "qed":
            lines += [f"set etal {ETA_10DEG:.6f}", f"set etaa {ETA_10DEG:.6f}"]
        lines += ["done"]
    card = os.path.join(work, f"{pid}.mg5")
    with open(card, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    log = os.path.join(work, f"{pid}.log")
    with open(log, "w", encoding="utf-8") as handle:
        subprocess.run([MG5, card], cwd=work, stdout=handle, stderr=subprocess.STDOUT, check=False)
    text = open(log, encoding="utf-8", errors="replace").read()
    sigmas = [(float(a), float(b)) for a, b in re.findall(r"Cross-section :\s+([0-9.eE+-]+)\s+\+-\s+([0-9.eE+-]+) pb", text)]
    if len(sigmas) != len(energies):
        raise RuntimeError(f"{pid}: expected {len(energies)} MadGraph results, found {len(sigmas)}; see {log}")
    result = {}
    for index, (energy, (sigma, error)) in enumerate(zip(energies, sigmas)):
        entry = {"sigma_pb": sigma, "error_pb": error}
        if pid == "ff13":
            entry["afb"] = forward_backward(os.path.join(out, "Events", f"run_{index + 1:02d}", "unweighted_events.lhe.gz"))
        result[f"{energy:.4f}"] = entry
    return result


def forward_backward(path: str) -> list[float]:
    """A_FB and its binomial error: the μ⁻ direction relative to the incoming e⁻ (read per event)."""
    import gzip

    forward = backward = 0
    electron_pz = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 13:
                continue
            if fields[0] == "11" and fields[1] == "-1":
                electron_pz = float(fields[8])
            elif fields[0] == "13" and fields[1] == "1":
                if float(fields[8]) * electron_pz > 0.0:
                    forward += 1
                else:
                    backward += 1
    n = forward + backward
    afb = (forward - backward) / n
    return [afb, math.sqrt((1.0 - afb * afb) / n)]


def smlab_values(pid: str, energy: float, matched: bool, scheme: str) -> dict:
    """SMLab's Born σ (no ISR) and, for μμ, A_FB; optionally in MadGraph's coupling scheme."""
    process = process_by_id(pid)
    saved = (EW.alpha_em, EW.SIN2_THETA_W, EW.qcd_factor)
    if matched and scheme == "ew":
        EW.alpha_em = lambda s: ALPHA_GF
        EW.SIN2_THETA_W = SIN2_OS
        EW.qcd_factor = lambda pdg, sqrt_s: 1.0
    try:
        sigma = sigma_pb(process, BEAMS["ee"], energy, isr=False)
        out = {"sigma_pb": sigma}
        if pid == "ff13":
            a_v, a_a, a_1, beta = EW.fermion_pair_amplitudes(energy, 11, 13)
            # ∫ f over forward minus backward, divided by the total: A_FB = (3/8) A_1 / (A_V + A_A) for β → 1.
            out["afb"] = 0.375 * a_1 / (a_v + a_a)
        return out
    finally:
        EW.alpha_em, EW.SIN2_THETA_W, EW.qcd_factor = saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=20000)
    args = parser.parse_args()
    work = tempfile.mkdtemp(prefix="smlab-mg5-compare-")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(run_madgraph, pid, proc, energies, scheme, args.events, work): pid
                       for pid, proc, energies, scheme in CASES}
            madgraph = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows = []
    for pid, proc, energies, scheme in CASES:
        for energy in energies:
            mg = madgraph[pid][f"{energy:.4f}"]
            matched = smlab_values(pid, energy, True, scheme)
            shipped = smlab_values(pid, energy, False, scheme)
            row = {
                "process": pid, "madgraph_process": proc, "sqrt_s": energy, "scheme": scheme,
                "madgraph_pb": mg["sigma_pb"], "madgraph_err_pb": mg["error_pb"],
                "smlab_matched_pb": matched["sigma_pb"], "smlab_shipped_pb": shipped["sigma_pb"],
                "matched_ratio": matched["sigma_pb"] / mg["sigma_pb"],
                "shipped_ratio": shipped["sigma_pb"] / mg["sigma_pb"],
            }
            if "afb" in mg:
                row.update(madgraph_afb=mg["afb"][0], madgraph_afb_err=mg["afb"][1],
                           smlab_matched_afb=matched["afb"], smlab_shipped_afb=shipped["afb"])
            rows.append(row)
            print(f"{pid:9s} {energy:7.2f} GeV  MG {mg['sigma_pb']:11.5g} ± {mg['error_pb']:.2g} pb   "
                  f"SMLab matched {row['matched_ratio']:.4f}   shipped {row['shipped_ratio']:.4f}", flush=True)
    with open(os.path.join(HERE, "builtin_vs_madgraph.json"), "w", encoding="utf-8") as handle:
        json.dump({"sin2_on_shell": SIN2_OS, "alpha_gf": ALPHA_GF, "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
