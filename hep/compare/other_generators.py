"""SMLab's validation benchmarks run with the other general-purpose generators, Sherpa and Herwig.

    ~/micromamba/envs/smlab-hep/bin/python hep/compare/other_generators.py sherpa|herwig [--scale 1.0]

Each generator writes HepMC3 events into a named pipe that SMLab's Rivet
reads, in parallel chunks with independent seeds; the chunks are merged with
``rivet-merge -e`` and scored against the data with exactly the code of
hep/validate.py, so the numbers sit directly beside SMLab's. The generator
setups follow the authors' own example run cards (Sherpa 3.0 Examples/,
Herwig 7.3 share/Herwig/), changing only the beams, the energy, and the event
output. References: Sherpa 3 (E. Bothmann et al., JHEP 12 (2024) 156,
arXiv:2410.22148); Herwig 7.3 (G. Bewick et al., Eur. Phys. J. C 84 (2024)
1053, arXiv:2312.05175). Both run at leading order with parton showers (Sherpa merges extra
jets, MEPS@LO/CKKW): their NLO modes need OpenLoops, which is not installed.
Writes hep/compare/<generator>.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HEP = os.path.dirname(HERE)
sys.path.insert(0, HEP)

import validate  # noqa: E402

RIVET_BIN = os.path.join(sys.prefix, "bin")
SHERPA = os.path.expanduser("~/micromamba/envs/smlab-sherpa/bin/Sherpa")
HERWIG_PREFIX = os.path.expanduser("~/herwig/install")

# Generator runs: (run key, benchmarks it serves, events). The t t̄ sample feeds both t t̄ analyses.
RUNS = [
    ("lep", ["lep_z_hadrons"], 60000),
    ("minbias", ["lhc_minbias"], 150000),
    ("z", ["lhc_z_pt"], 150000),
    ("ttbar", ["lhc_ttbar", "lhc_ttbar_dilep"], 60000),
    ("jets", ["lhc_jets"], 150000),
]

SHERPA_CARDS = {
    # Examples/Jets_at_LeptonColliders/LEP_Jets, without initial-state radiation (as the ALEPH data and SMLab),
    # and with LEP's stable-particle convention: τ > 1 ns (cτ > 300 mm) stable, so K⁰_S and Λ decay; ALEPH
    # counted their charged products. (Sherpa's default, cτ > 10 mm, leaves both undecayed.)
    "lep": """HADRON_DECAYS: {Max_Proper_Lifetime: 300.}
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
PDF_LIBRARY: None
YFS: {MODE: None}
ALPHAS(MZ): 0.1188
ORDER_ALPHAS: 1
PROCESSES:
- 11 -11 -> 93 93 93{3}:
    CKKW: pow(10,-2.25/2.00)*E_CMS
    Order: {QCD: 0, EW: 2}
""",
    # Examples/Soft_QCD/LHC_7TeV_MinBias/Amisic.yaml at 13 TeV, its built-in Rivet call removed.
    "minbias": """BEAMS: 2212
BEAM_ENERGIES: 6500
EVENT_TYPE: MinimumBias
ME_GENERATORS: None
SOFT_COLLISIONS: Amisic
YFS: {MODE: None}
REMNANTS:
  2212: {MATTER_FORM: Single_Gaussian, MATTER_RADIUS1: 1., MATTER_RADIUS2: 0., MATTER_FRACTION1: 1.}
COLOUR_RECONNECTIONS: {MODE: On, PMODE: Log, Q_0: 1., ETA_Q: 0.1, RESHUFFLE: 0.1111, KAPPA: 1.}
AHADIC:
  KT_0: 1.2
  PT_MAX: 0.68
  ALPHA_G: 1.
  ALPHA_L: 3.9
  BETA_L: 0.18
  GAMMA_L: 0.48
  ALPHA_D: 3.4
  BETA_D: 0.72
  GAMMA_D: 0.77
  ALPHA_H: -0.6
  BETA_H: 1.8
  GAMMA_H: 0.024
  ALPHA_B: 14.2
  BETA_B: 1.8
  GAMMA_B: 8.1
  DECAY_THRESHOLD: 0.02
  STRANGE_FRACTION: 0.46
  BARYON_FRACTION: 0.17
  P_QS_by_PQQ_norm: 0.056
  P_QQ1_by_PQQ0: 0.60
""",
    # Examples/V_plus_Jets/LHC_ZJets at LO: Z + 0, 1, 2 jets merged (MEPS@LO), e and μ, m_ℓℓ > 60 GeV.
    "z": """BEAMS: 2212
BEAM_ENERGIES: 6500
PROCESSES:
- 93 93 -> 11 -11 93{2}:
    Order: {QCD: 0, EW: 2}
    CKKW: 20
- 93 93 -> 13 -13 93{2}:
    Order: {QCD: 0, EW: 2}
    CKKW: 20
SELECTORS:
- [Mass, 11, -11, 60, E_CMS]
- [Mass, 13, -13, 60, E_CMS]
""",
    # Examples/Tops_plus_Jets/LHC_Tops at LO: t t̄ + 0, 1 jet merged, every top decay channel open.
    "ttbar": """BEAMS: 2212
BEAM_ENERGIES: 6500
EXCLUSIVE_CLUSTER_MODE: 1
MEPS: {CORE_SCALE: TTBar}
HARD_DECAYS: {Enabled: true}
PARTICLE_DATA:
  6: {Width: 0}
PROCESSES:
- 93 93 -> 6 -6 93{1}:
    Order: {QCD: 2, EW: 0}
    CKKW: 20
""",
    # Examples/Jets_at_HadronColliders/LHC_Jets_MEPS: 2 → 2, jets above 100 GeV (as SMLab's p̂T > 100 GeV).
    "jets": """BEAMS: 2212
BEAM_ENERGIES: 6500
PROCESSES:
- 93 93 -> 93 93:
    Order: {QCD: 2, EW: 0}
    Integration_Error: 0.02
SELECTORS:
- NJetFinder: {N: 2, PTMin: 100.0, ETMin: 0.0, R: 0.4, Exp: -1}
""",
}


def _analyses(run_key: str) -> list[str]:
    benches = next(b for k, b, _n in RUNS if k == run_key)
    return [a for bench in validate.BENCHMARKS if bench["key"] in benches for a in bench["analyses"]]


def _rivet_env() -> dict:
    return dict(os.environ, PATH=RIVET_BIN + os.pathsep + os.environ.get("PATH", ""))


# -- Sherpa ------------------------------------------------------------------------

def sherpa_prepare(run_key: str, base: str) -> None:
    """Initialize and integrate once; the chunks copy the results."""
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "Sherpa.yaml"), "w", encoding="utf-8") as handle:
        handle.write(SHERPA_CARDS[run_key] + "EVENTS: 0\n")
    with open(os.path.join(base, "init.log"), "w", encoding="utf-8") as log:
        subprocess.run([SHERPA], cwd=base, stdout=log, stderr=subprocess.STDOUT, check=True)


def sherpa_chunk(run_key: str, base: str, directory: str, n: int, seed: int) -> list[str]:
    shutil.copytree(base, directory, ignore=shutil.ignore_patterns("*.log"))
    with open(os.path.join(directory, "Sherpa.yaml"), "w", encoding="utf-8") as handle:
        handle.write(SHERPA_CARDS[run_key]
                     + f"EVENTS: {n}\nRANDOM_SEED: {seed}\nEVENT_OUTPUT: HepMC3_GenEvent[events]\n")
    return [SHERPA]


# -- Herwig ------------------------------------------------------------------------

# Herwig 7.3's own inputs (share/Herwig/LEP.in, LHC-MB.in, LHC.in) with its HepMC snippet;
# only the energy, the hard process, and generation cuts matching SMLab's are set here.
HERWIG_INPUTS = {
    # No initial-state radiation, as ALEPH's data are corrected for it: EECollider.in radiates photons off
    # the beams both in the shower and through electron structure functions (beam photons in 263 of 300
    # events). DoISR No removes the former, NoPDF on e± the latter (verified: 0 beam photons in 300
    # events). Final-state QED and QCD radiation are unaffected.
    "lep": """read snippets/EECollider.in
set /Herwig/Shower/ShowerHandler:DoISR No
set /Herwig/Particles/e-:PDF /Herwig/Partons/NoPDF
set /Herwig/Particles/e+:PDF /Herwig/Partons/NoPDF
cd /Herwig/MatrixElements
insert SubProcess:MatrixElements 0 MEee2gZ2qq
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 91.2
""",
    "minbias": """read snippets/PPCollider.in
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 13000.0
set /Herwig/Shower/ShowerHandler:IntrinsicPtGaussian 2.2*GeV
read snippets/MB.in
read snippets/Diffraction.in
""",
    "z": """read snippets/PPCollider.in
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 13000.0
cd /Herwig/MatrixElements/
insert SubProcess:MatrixElements[0] MEqq2gZ2ff
set MEqq2gZ2ff:Process ChargedLeptons
set /Herwig/Cuts/MassCut:MinM 60.*GeV
""",
    "ttbar": """read snippets/PPCollider.in
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 13000.0
cd /Herwig/MatrixElements/
insert SubProcess:MatrixElements[0] MEHeavyQuark
""",
    "jets": """read snippets/PPCollider.in
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 13000.0
cd /Herwig/MatrixElements/
insert SubProcess:MatrixElements[0] MEQCD2to2
set /Herwig/Cuts/JetKtCut:MinKT 100.*GeV
""",
}


def _herwig_env() -> dict:
    lib = os.path.join(HERWIG_PREFIX, "lib")
    return dict(os.environ,
                PATH=os.path.join(HERWIG_PREFIX, "bin") + os.pathsep + os.environ.get("PATH", ""),
                LD_LIBRARY_PATH=os.pathsep.join([lib, os.path.join(lib, "ThePEG"), os.path.join(lib, "Herwig")]),
                LHAPDF_DATA_PATH=os.path.join(HERWIG_PREFIX, "share", "LHAPDF"))


def herwig_prepare(run_key: str, base: str) -> None:
    """Build the run file once (Herwig's 'read' step); the chunks copy it."""
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "run.in"), "w", encoding="utf-8") as handle:
        handle.write(HERWIG_INPUTS[run_key]
                     + "read snippets/HepMC.in\nset /Herwig/Analysis/HepMC:Filename events\n"
                     + "set /Herwig/Analysis/HepMC:PrintEvent 100000000\n"
                     + "cd /Herwig/Generators\nsaverun smlab EventGenerator\n")
    with open(os.path.join(base, "read.log"), "w", encoding="utf-8") as log:
        subprocess.run([os.path.join(HERWIG_PREFIX, "bin", "Herwig"), "read", "run.in"], cwd=base,
                       env=_herwig_env(), stdout=log, stderr=subprocess.STDOUT, check=True)


def herwig_chunk(run_key: str, base: str, directory: str, n: int, seed: int) -> list[str]:
    shutil.copytree(base, directory, ignore=shutil.ignore_patterns("*.log"))
    return [os.path.join(HERWIG_PREFIX, "bin", "Herwig"), "run", "smlab.run", "-N", str(n), "-s", str(seed), "-q"]


GENERATORS = {"sherpa": (sherpa_prepare, sherpa_chunk), "herwig": (herwig_prepare, herwig_chunk)}


def _chunk(args) -> str:
    generator, run_key, base, work, index, n = args
    directory = os.path.join(work, f"chunk{index}")
    command = GENERATORS[generator][1](run_key, base, directory, n, 1000 + 7919 * index)
    fifo = os.path.join(directory, "events")
    os.mkfifo(fifo)
    out = os.path.join(work, f"chunk{index}.yoda")
    rivet = subprocess.Popen([os.path.join(RIVET_BIN, "rivet"), "-q", "-a", ",".join(_analyses(run_key)), "-o", out, fifo],
                             env=_rivet_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    env = _herwig_env() if generator == "herwig" else None
    with open(os.path.join(directory, "run.log"), "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    _out, err = rivet.communicate()
    if rivet.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"rivet failed on {run_key} chunk {index}: {err[-400:]}")
    shutil.rmtree(directory, ignore_errors=True)
    return out


def run(generator: str, run_key: str, events: int, work: str) -> str:
    base = os.path.join(work, "base")
    GENERATORS[generator][0](run_key, base)
    workers = max(1, min(mp.cpu_count() - 2, 30))
    jobs = [(generator, run_key, base, work, i, n) for i, (n, _o) in enumerate(validate.chunk_plan(events, workers))]
    with mp.get_context("fork").Pool(workers) as pool:
        outputs = pool.map(_chunk, jobs)
    merged = os.path.join(HERE, "yoda", f"{generator}_{run_key}.yoda")
    os.makedirs(os.path.dirname(merged), exist_ok=True)
    subprocess.run([os.path.join(RIVET_BIN, "rivet-merge"), "-e", "-o", merged, *outputs], check=True,
                   env=_rivet_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generator", choices=sorted(GENERATORS))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", default=None, help="run keys: lep minbias z ttbar jets")
    args = parser.parse_args()
    path = os.path.join(HERE, f"{args.generator}.json")
    report = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for run_key, benches, events in RUNS:
        if args.only and run_key not in args.only:
            continue
        started = time.time()
        with tempfile.TemporaryDirectory(prefix=f"smlab-{args.generator}-{run_key}-") as work:
            merged = run(args.generator, run_key, max(200, int(events * args.scale)), work)
        for bench in validate.BENCHMARKS:
            if bench["key"] not in benches:
                continue
            plots = validate.compare(merged, bench["analyses"])
            chi = [p["chi2_ndf"] for p in plots]
            report[bench["key"]] = {
                "generator": args.generator, "events": int(events * args.scale), "seconds": round(time.time() - started),
                "median_chi2_ndf": validate._median(chi),
                "median_shape_chi2_ndf": validate._median([p["shape_chi2_ndf"] for p in plots]),
                "median_norm_ratio": validate._median([p["norm_ratio"] for p in plots]),
                "plots": plots,
            }
            print(f"{args.generator} {bench['key']:16s} χ²/ndf {report[bench['key']]['median_chi2_ndf']:.2f}  "
                  f"shape {report[bench['key']]['median_shape_chi2_ndf']:.2f}  "
                  f"rate {report[bench['key']]['median_norm_ratio']:.2f}  ({len(plots)} plots)", flush=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle)


if __name__ == "__main__":
    main()
