"""Validate SMLab's PYTHIA (and MadGraph) events against published data with Rivet.

    ~/micromamba/envs/smlab-hep/bin/python hep/validate.py [--quick] [--only KEY ...]

Each benchmark generates events in parallel chunks (one per CPU core, each with
its own seed), streams them as HepMC3 through a named pipe into Rivet, merges
the chunks with ``rivet-merge``, and compares every histogram that has
reference data with the measurement shipped with Rivet:

    χ²/ndf = (1/N) Σ_bins (MC − data)² / (σ_data² + σ_MC²)

Results go to hep/validation/results.json (read by the app's Validation
panel) and the merged YODA files to hep/validation/<key>.yoda.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BIN = os.path.join(sys.prefix, "bin")
OUT_DIR = os.path.join(HERE, "validation")

# Each benchmark: which physics it tests, the SMLab configuration, the Rivet
# analysis (INSPIRE-coded name), and the number of events.
BENCHMARKS = [
    {
        "key": "lep_z_hadrons",
        "title": "LEP, Z → hadrons at 91.2 GeV",
        "tests": "Final-state shower and Lund hadronization",
        "analyses": ["ALEPH_1996_I428072"],
        "reference": "ALEPH, Phys. Rept. 294 (1998) 1",
        "config": {"beams": "ee", "process": "ll_gmz_had", "sqrt_s": 91.2, "isr": False},
        "events": 60000,
    },
    {
        "key": "lhc_minbias",
        "title": "LHC 13 TeV, minimum-bias charged particles",
        "tests": "Multiparton interactions and soft QCD",
        "analyses": ["ATLAS_2016_I1419652"],
        "reference": "ATLAS, Phys. Lett. B 758 (2016) 67, arXiv:1602.01633",
        "config": {"beams": "pp", "process": "pp_minbias", "sqrt_s": 13000.0},
        "events": 150000,
    },
    {
        "key": "lhc_z_pt",
        "title": "LHC 13 TeV, Z boson transverse momentum",
        "tests": "Initial-state radiation and the Drell–Yan hard process",
        "analyses": ["ATLAS_2019_I1768911"],
        "reference": "ATLAS, Eur. Phys. J. C 80 (2020) 616",
        "config": {"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13000.0},
        "events": 150000,
        "variants": [
            {"suffix": "nlo", "label": "MadGraph5_aMC@NLO + PYTHIA 8.3", "source": {"source": "nlo", "sample": "dy"}},
            {"suffix": "fxfx", "label": "FxFx Z+0,1,2j NLO + PYTHIA 8.3", "source": {"source": "nlo", "sample": "dy_fxfx"}},
            {"suffix": "fxfx_tuned", "label": "FxFx + PYTHIA 8.3, SMLab tune",
             "source": {"source": "nlo", "sample": "dy_fxfx"}, "tuned": True},
        ],
    },
    {
        "key": "lhc_ttbar",
        "title": "LHC 13 TeV, t t̄ event variables (lepton + jets)",
        "tests": "Top-quark production, decay, and radiation",
        "analyses": ["CMS_2018_I1662081"],
        "reference": "CMS, JHEP 06 (2018) 002, arXiv:1803.03991",
        "config": {"beams": "pp", "process": "pp_ttbar", "sqrt_s": 13000.0},
        "events": 60000,
        "variants": [
            {"suffix": "nlo", "label": "MadGraph5_aMC@NLO + PYTHIA 8.3", "source": {"source": "nlo", "sample": "ttbar"}},
            {"suffix": "nlo_ms", "label": "aMC@NLO + MadSpin + PYTHIA 8.3", "source": {"source": "nlo", "sample": "ttbar_ms"}},
            {"suffix": "ms_tuned", "label": "aMC@NLO + MadSpin, SMLab tune",
             "source": {"source": "nlo", "sample": "ttbar_ms"}, "tuned": True},
        ],
    },
    {
        "key": "lhc_ttbar_dilep",
        "title": "LHC 13 TeV, dileptonic t t̄ (eμ), incl. Δφ(e, μ)",
        "tests": "Top-quark spin correlations and lepton kinematics",
        "analyses": ["ATLAS_2019_I1759875"],
        "reference": "ATLAS, Eur. Phys. J. C 80 (2020) 528",
        "config": {"beams": "pp", "process": "pp_ttbar", "sqrt_s": 13000.0},
        "events": 60000,
        "variants": [
            {"suffix": "nlo", "label": "MadGraph5_aMC@NLO + PYTHIA 8.3", "source": {"source": "nlo", "sample": "ttbar"}},
            {"suffix": "nlo_ms", "label": "aMC@NLO + MadSpin + PYTHIA 8.3", "source": {"source": "nlo", "sample": "ttbar_ms"}},
            {"suffix": "ms_tuned", "label": "aMC@NLO + MadSpin, SMLab tune",
             "source": {"source": "nlo", "sample": "ttbar_ms"}, "tuned": True},
        ],
    },
    {
        "key": "lhc_jets",
        "title": "LHC 13 TeV, inclusive jet cross section",
        "tests": "QCD 2 → 2 at leading order, absolute normalization",
        "analyses": ["CMS_2016_I1459051"],
        "reference": "CMS, Eur. Phys. J. C 76 (2016) 451",
        "config": {"beams": "pp", "process": "pp_dijet", "sqrt_s": 13000.0},
        "events": 150000,
    },
]


def _write_events(config: dict, seed: int, n: int, fifo: str, source: dict | None) -> dict:
    """Generate ``n`` events and write them as HepMC3 into ``fifo``. Returns run info."""
    import pyhepmc as hep

    from detector import to_hepmc
    from worker import Worker

    worker = Worker()
    worker.init({**config, "seed": seed, "decays": "generator", **(source or {})})
    run_info = hep.GenRunInfo()
    run_info.weight_names = ["Default"]
    done = 0
    sigma = err = 0.0
    with hep.io.WriterAscii(fifo) as writer:
        while done < n:
            batch = min(500, n - done)
            reply = worker.generate(batch, jets=False)
            sigma, err = reply["sigma_pb"], reply["sigma_err_pb"]
            if reply.get("nlo") and reply["nlo"].get("sigma_pb") and not reply["nlo"].get("fxfx"):
                # MC@NLO: the NLO cross section of the whole sample, not PYTHIA's running estimate.
                sigma, err = reply["nlo"]["sigma_pb"], reply["nlo"].get("sigma_err_pb", 0.0)
            for raw in reply["events"]:
                event = to_hepmc(raw["particles"], done)
                event.run_info = run_info
                # MC@NLO events carry signed weights; Rivet must see the sign.
                event.weights = [float(raw.get("weight", 1.0))]
                xs = hep.GenCrossSection()
                xs.set_cross_section(sigma, err, done + 1, done + 1)
                event.cross_section = xs
                writer.write(event)
                done += 1
    return {"sigma_pb": sigma, "sigma_err_pb": err, "events": done}


def _chunk(args) -> dict:
    bench, index, n, work, source = args
    fifo = os.path.join(work, f"chunk{index}.hepmc")
    yoda_out = os.path.join(work, f"chunk{index}.yoda")
    os.mkfifo(fifo)
    rivet = subprocess.Popen(
        [os.path.join(BIN, "rivet"), "-q", "-a", ",".join(bench["analyses"]), "-o", yoda_out, fifo],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    info = _write_events(bench["config"], 7919 * (index + 1), n, fifo, source)
    _out, err = rivet.communicate()
    if rivet.returncode != 0 or not os.path.exists(yoda_out):
        raise RuntimeError(f"rivet failed on chunk {index}: {err[-500:]}")
    info["yoda"] = yoda_out
    return info


def _estimate(ao):
    """(edges, values, errors) of a YODA 2 histogram or estimate, or None."""
    kind = type(ao).__name__
    if kind.startswith("BinnedHisto") or kind.startswith("Histo1D"):
        ao = ao.mkEstimate()
        kind = type(ao).__name__
    if not (kind.startswith("BinnedEstimate1D") or kind.startswith("Estimate1D")):
        return None
    if ao.binDim() != 1:
        return None
    edges = list(ao.xEdges())
    values, errors = [], []
    for b in ao.bins():
        values.append(float(b.val()))
        # Rivet 4 names its uncertainty sources ("stats" for MC; "stat", "sys", or
        # correlated/uncorrelated for data). The total combines them in quadrature.
        errors.append(float(b.totalErrAvg()))
    if len(values) != len(edges) - 1:
        return None
    return edges, values, errors


def plot_labels(analysis: str) -> list[tuple[str, dict]]:
    """Title and axis labels from the analysis .plot file (BEGIN PLOT blocks; paths may be regexes)."""
    import re

    import rivet

    try:
        path = rivet.findAnalysisPlotFile(f"{analysis}.plot")
    except Exception:
        return []
    if not path or not os.path.exists(path):
        return []
    blocks: list[tuple[str, dict]] = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("# BEGIN PLOT"):
                current = (line.split("PLOT", 1)[1].strip(), {})
            elif line.startswith("# END PLOT"):
                if current:
                    blocks.append(current)
                current = None
            elif current and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() in ("Title", "XLabel", "YLabel", "LogX", "LogY"):
                    current[1][key.strip()] = value.strip()
    compiled = []
    for pattern, values in blocks:
        try:
            compiled.append((re.compile(pattern), values))
        except re.error:
            continue
    return compiled


def _labels_for(path: str, compiled) -> dict:
    merged: dict = {}
    for regex, values in compiled:
        if regex.fullmatch(path) or regex.match(path):
            merged.update(values)
    return merged


def compare(yoda_file: str, analyses: list[str]) -> list[dict]:
    import rivet
    import yoda

    mc = yoda.read(yoda_file)
    out = []
    for analysis in analyses:
        ref_path = rivet.findAnalysisRefFile(f"{analysis}.yoda")
        refs = yoda.read(ref_path)
        labels = plot_labels(analysis)
        for path, ref in sorted(refs.items()):
            key = path.replace("/REF", "", 1)
            if key not in mc:
                continue
            data = _estimate(ref)
            model = _estimate(mc[key])
            if data is None or model is None or len(data[1]) != len(model[1]):
                continue
            widths = [b - a for a, b in zip(data[0][:-1], data[0][1:])]
            usable = [
                (d, e, m, me, w)
                for d, e, m, me, w in zip(data[1], data[2], model[1], model[2], widths)
                if math.isfinite(d) and math.isfinite(m) and e * e + me * me > 0.0 and not (d == 0.0 and m == 0.0)
            ]
            ndf = len(usable)
            if ndf == 0 or sum(abs(v) for v in model[1]) == 0.0:
                continue
            chi2 = sum((m - d) ** 2 / (e * e + me * me) for d, e, m, me, _w in usable)
            # Normalization and shape separately: scale MC to the data integral.
            data_area = sum(d * w for d, _e, _m, _me, w in usable)
            mc_area = sum(m * w for _d, _e, m, _me, w in usable)
            ratio = mc_area / data_area if data_area else float("nan")
            if ratio and math.isfinite(ratio):
                shape = sum(
                    (m / ratio - d) ** 2 / (e * e + (me / ratio) ** 2) for d, e, m, me, _w in usable
                ) / ndf
            else:
                shape = float("nan")
            meta = _labels_for(key, labels)
            out.append({
                "path": key,
                "title": meta.get("Title") or ref.title() or key,
                "xlabel": meta.get("XLabel", ""),
                "ylabel": meta.get("YLabel", ""),
                "chi2_ndf": chi2 / ndf,
                "shape_chi2_ndf": shape,
                "norm_ratio": ratio,
                "ndf": ndf,
                "edges": data[0],
                "data": data[1],
                "data_err": data[2],
                "mc": model[1],
                "mc_err": model[2],
            })
    return out


# FxFx merging vetoes 55–56 % of the LHE events (measured: 2000 accepted of 4502
# read in the 13 TeV Z + 0, 1, 2 jet sample). Each chunk then reads more of the
# file than it keeps, so chunks start at evenly spaced LHE positions and the
# usable number of events is capped below the measured acceptance.
FXFX_USABLE_FRACTION = 0.40


def sample_budget(sample: dict) -> tuple[int, int]:
    """(usable showered events, LHE events) of an MC@NLO sample."""
    lhe = int(sample["events"])
    return (int(lhe * FXFX_USABLE_FRACTION) if sample.get("fxfx") else lhe), lhe


def chunk_plan(n_total: int, workers: int, lhe_events: int | None = None) -> list[tuple[int, int]]:
    """(events, first LHE event) for each parallel chunk.

    Without vetoes each chunk reads exactly the events it keeps, so chunks follow
    one another. With FxFx (``lhe_events`` given) a chunk reads more than it keeps,
    so chunks start at evenly spaced positions across the whole file.
    """
    per = [n_total // workers + (1 if i < n_total % workers else 0) for i in range(workers)]
    if lhe_events is not None:
        offsets = [i * lhe_events // workers for i in range(workers)]
    else:
        offsets = [sum(per[:i]) for i in range(workers)]
    return [(n, offset) for n, offset in zip(per, offsets) if n > 0]


def run(bench: dict, scale: float, source: dict | None = None, label: str = "PYTHIA 8.3 LO+PS",
        available: int | None = None, variant_suffix: str = "", lhe_events: int | None = None) -> dict:
    n_total = max(200, int(bench["events"] * scale))
    if available is not None:
        # An MC@NLO sample is finite: use each of its events exactly once.
        n_total = min(n_total, available)
    workers = max(1, min(mp.cpu_count() - 2, 30))
    plan = chunk_plan(n_total, workers, lhe_events)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"smlab-rivet-{bench['key']}-") as work:
        jobs = []
        for i, (n, offset) in enumerate(plan):
            chunk_source = {**source, "lhef_skip": offset} if source else None
            jobs.append((bench, i, n, work, chunk_source))
        with mp.get_context("fork").Pool(workers) as pool:
            infos = pool.map(_chunk, jobs)
        suffix = f"_{variant_suffix}" if variant_suffix else ""
        merged = os.path.join(OUT_DIR, f"{bench['key']}{suffix}.yoda")
        subprocess.run(
            [os.path.join(BIN, "rivet-merge"), "-e", "-o", merged] + [info["yoda"] for info in infos],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    plots = compare(merged, bench["analyses"])
    chi = [p["chi2_ndf"] for p in plots]
    sigma = sum(info["sigma_pb"] for info in infos) / len(infos)
    return {
        "key": bench["key"] + (f"_{variant_suffix}" if variant_suffix else ""),
        "variant": variant_suffix or "lo",
        "benchmark": bench["key"],
        "generator": label,
        "title": bench["title"],
        "tests": bench["tests"],
        "analyses": bench["analyses"],
        "reference": bench["reference"],
        "config": bench["config"],
        "events": sum(info["events"] for info in infos),
        "sigma_pb": sigma,
        "seconds": time.time() - started,
        "n_plots": len(plots),
        "median_chi2_ndf": sorted(chi)[len(chi) // 2] if chi else None,
        "median_shape_chi2_ndf": _median([p["shape_chi2_ndf"] for p in plots]),
        "median_norm_ratio": _median([p["norm_ratio"] for p in plots]),
        "plots": plots,
    }


def load_tune() -> dict | None:
    """The SMLab shower tune written by hep/tune.py, if one exists."""
    path = os.path.join(HERE, "validation", "tune.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _median(values: list[float]) -> float | None:
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    return finite[len(finite) // 2] if finite else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 % of the events, to test the pipeline")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-nlo", action="store_true", help="skip the MadGraph5_aMC@NLO variants")
    parser.add_argument("--nlo-only", action="store_true", help="run only the MadGraph5_aMC@NLO variants")
    parser.add_argument("--variants", nargs="*", default=None, help="variant suffixes to run (lo, nlo, fxfx, ...)")
    parser.add_argument("--out", default=None, help="directory for results.json (default hep/validation)")
    args = parser.parse_args()
    global OUT_DIR
    if args.out:
        OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")
    results_path = os.path.join(OUT_DIR, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as handle:
            results = json.load(handle).get("benchmarks", {})
    scale = 0.02 if args.quick else 1.0
    from worker import nlo_samples

    samples = nlo_samples(all_variants=True)
    tune = load_tune()
    for bench in BENCHMARKS:
        if args.only and bench["key"] not in args.only:
            continue
        runs = [("lo", None, "PYTHIA 8.3 LO+PS", None)]
        energy_key = f"{bench['config']['process']}@{int(bench['config']['sqrt_s'])}"
        for variant in bench.get("variants", []):
            sample = samples.get(f"{energy_key}#{variant['source']['sample']}")
            if sample is None or args.no_nlo:
                continue
            source = dict(variant["source"])
            label = variant["label"]
            if variant.get("tuned"):
                if not tune:
                    continue
                source["tune"] = tune["settings"]
            runs.append((variant["suffix"], source, label, sample_budget(sample)))
        for suffix, source, label, budget in runs:
            if args.variants and suffix not in args.variants:
                continue
            if args.nlo_only and source is None:
                continue
            available, lhe_events = budget or (None, None)
            result = run(bench, scale, source, label, available, "" if suffix == "lo" else suffix, lhe_events)
            if source:
                sample = samples[f"{energy_key}#{source['sample']}"]
                # For FxFx, MadGraph's σ is before merging; the result's σ (PYTHIA, after the veto) is the physical one.
                result["nlo_sigma_pb"] = None if sample.get("fxfx") else sample.get("sigma_pb")
                result["nlo_scale_pct"] = [sample.get("scale_up_pct"), sample.get("scale_down_pct")]
                if source.get("tune"):
                    result["tune"] = source["tune"]
            results[result["key"]] = result
            _report(result, results, results_path)
    shutil.rmtree(os.path.join(HERE, "__pycache__"), ignore_errors=True)


def _report(result: dict, results: dict, results_path: str) -> None:
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"{result['key']:19s} {result['events']:>7d} ev  {result['n_plots']:>3d} plots  "
          f"χ²/ndf {fmt(result['median_chi2_ndf'])}  shape {fmt(result['median_shape_chi2_ndf'])}  "
          f"MC/data {fmt(result['median_norm_ratio'])}  σ {result['sigma_pb']:.4g} pb  {result['seconds']:.0f}s", flush=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "benchmarks": results}, handle)

if __name__ == "__main__":
    main()
