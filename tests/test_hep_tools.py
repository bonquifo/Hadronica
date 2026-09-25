"""The research-grade toolchain, checked against independent references.

Every assertion here compares with something outside the code under test: an
exact expectation built into synthetic input, the PDG 2026 values in
smlab/constants.py, an independent recomputation of stored results, or the
numbers quoted in the documentation. None of these tests needs WSL.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys

import pytest

from smlab import constants as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEP = os.path.join(ROOT, "hep")
sys.path.insert(0, HEP)

import tune  # noqa: E402
import validate  # noqa: E402

RESULTS = json.load(open(os.path.join(HEP, "validation", "results.json"), encoding="utf-8"))["benchmarks"]
TUNE = json.load(open(os.path.join(HEP, "validation", "tune.json"), encoding="utf-8"))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


# -- Fortran '$' formats (hep/mg5/fix_dollar_formats.py) ------------------------

FIXER = os.path.join(HEP, "mg5", "fix_dollar_formats.py")


def _fix(tmp_path, source: str) -> tuple[str, str]:
    path = tmp_path / "t.f"
    path.write_text(source, encoding="latin-1")
    done = subprocess.run([sys.executable, FIXER, str(path)], capture_output=True, text=True, check=True)
    return path.read_text(encoding="latin-1"), done.stdout


def test_dollar_formats_become_non_advancing_writes(tmp_path):
    source = (
        "      write(*,'(a$)') ' '\n"
        "      WRITE(6, '(3x,a,$)') 'x'\n"
        "      write(lun,'(i5,1x,$)') n\n"
    )
    fixed, report = _fix(tmp_path, source)
    assert fixed.splitlines() == [
        "      write(*,'(a)',advance='no') ' '",
        "      write(6,'(3x,a)',advance='no') 'x'",
        "      write(lun,'(i5,1x)',advance='no') n",
    ]
    assert "3 replaced, 0 left" in report


def test_formats_built_at_run_time_are_fixed_too(tmp_path):
    # madevent_symmetry.f, which MadSpin compiles: the format string is assembled, then used.
    source = (
        "                  write(formstr,'(a,i1,a)') '(I',nconf,'$)'\n"
        "                  write(*,formstr) mapconfig(i)\n"
        "                  write(27,formstr2) mapconfig(i),use_config(i)\n"
    )
    fixed, _ = _fix(tmp_path, source)
    lines = fixed.splitlines()
    assert lines[0].endswith("'(I',nconf,')'")
    assert lines[1].strip() == "write(*,formstr,advance='no') mapconfig(i)"
    assert lines[2] == source.splitlines()[2]  # the file write keeps its own (advancing) format


def test_fixed_form_continuations_and_comments_are_untouched(tmp_path):
    source = (
        "      call foo(a,\n"
        "     $         b)\n"
        "c     write(*,'(a$)') 'commented out'\n"
    )
    fixed, report = _fix(tmp_path, source)
    assert fixed.splitlines()[1] == "     $         b)"
    assert "0 left" in report


def test_fixer_is_idempotent(tmp_path):
    once, _ = _fix(tmp_path, "      write(*,'(a$)') ' '\n")
    twice, report = _fix(tmp_path, once)
    assert once == twice and "0 replaced" in report


# -- LHE summary (hep/mg5/sample_info.py) ----------------------------------------

SCALE_FACTORS = [1.00, 1.08, 0.93, 1.05, 0.95, 1.12, 0.90, 1.02, 0.98]


def _lhe_event(weight: float) -> str:
    wgts = "".join(f"<wgt id='{1001 + i}'> {weight * f:+.10e} </wgt>\n" for i, f in enumerate(SCALE_FACTORS))
    return (
        "<event>\n"
        f" 3 0 {weight:+.10e} 9.1e+01 7.5e-03 1.2e-01\n"
        "       21 -1    0    0  501  502 +0.0e+00 +0.0e+00 +4.5e+01 4.5e+01 0.0e+00 0.0e+00 9.0e+00\n"
        "<rwgt>\n" + wgts + "</rwgt>\n</event>\n"
    )


def test_sample_summary_counts_signed_weights_and_the_scale_envelope(tmp_path):
    weights = [2.0, 2.0, -1.0, 1.0]  # mean 1.0, one negative in four
    sample = tmp_path / "s"
    sample.mkdir()
    (sample / "events.lhe").write_text("<LesHouchesEvents>\n" + "".join(map(_lhe_event, weights))
                                       + "</LesHouchesEvents>\n", encoding="utf-8")
    log = tmp_path / "mg5.log"
    log.write_text("INFO: Total cross section: 1.000e+00 +- 2.0e-02 pb\n", encoding="utf-8")
    subprocess.run([sys.executable, os.path.join(HEP, "mg5", "sample_info.py"), str(sample), "dy", "13000", "7",
                    str(log), '{"madspin": true}'], check=True, capture_output=True)
    info = json.loads((sample / "info.json").read_text(encoding="utf-8"))
    assert info["events"] == 4
    assert info["negative_fraction"] == pytest.approx(0.25)
    assert info["sigma_pb"] == pytest.approx(1.0) and info["sigma_err_pb"] == pytest.approx(0.02)
    assert info["sigma_from_events_pb"] == pytest.approx(1.0)
    assert info["scale_up_pct"] == pytest.approx(100 * (max(SCALE_FACTORS) - 1))
    assert info["scale_down_pct"] == pytest.approx(100 * (1 - min(SCALE_FACTORS)))
    assert info["madspin"] is True and info["sqrt_s"] == 13000.0


# -- MadSpin weight restoration (hep/mg5/restore_weights.py) ------------------------

RESTORE = os.path.join(HEP, "mg5", "restore_weights.py")


def _lhe(path, weights, factor=1.0):
    path.write_text("<LesHouchesEvents>\n" + "".join(_lhe_event(w * factor) for w in weights)
                    + "</LesHouchesEvents>\n", encoding="utf-8")


def _weights(path):
    text = path.read_text(encoding="utf-8")
    events = re.findall(r"<event>\n(.*?)</event>", text, re.S)
    return [(float(e.splitlines()[0].split()[2]), [float(v) for v in re.findall(r"<wgt id='\d+'> (\S+) </wgt>", e)])
            for e in events]


def test_madspin_weight_factor_is_removed_from_every_weight(tmp_path):
    weights = [2.0, -1.0, 1.5]
    undecayed, decayed = tmp_path / "u.lhe", tmp_path / "d.lhe"
    _lhe(undecayed, weights)
    _lhe(decayed, weights, factor=0.953832)
    done = subprocess.run([sys.executable, RESTORE, str(undecayed), str(decayed)], capture_output=True, text=True)
    assert done.returncode == 0 and float(done.stdout) == pytest.approx(0.953832)
    for (nominal, scales), weight in zip(_weights(decayed), weights):
        assert nominal == pytest.approx(weight, rel=1e-9)
        assert scales == pytest.approx([weight * f for f in SCALE_FACTORS], rel=1e-9)


def test_weights_changed_event_by_event_are_left_alone(tmp_path):
    undecayed, decayed = tmp_path / "u.lhe", tmp_path / "d.lhe"
    _lhe(undecayed, [2.0, 1.0])
    decayed.write_text("<LesHouchesEvents>\n" + _lhe_event(1.9) + _lhe_event(1.0) + "</LesHouchesEvents>\n",
                       encoding="utf-8")
    before = decayed.read_text(encoding="utf-8")
    done = subprocess.run([sys.executable, RESTORE, str(undecayed), str(decayed)], capture_output=True, text=True)
    assert done.returncode != 0 and "not scaled uniformly" in done.stderr
    assert decayed.read_text(encoding="utf-8") == before


# -- Chunking of finite MC@NLO samples (hep/validate.py) --------------------------

# FxFx vetoes measured on the samples: 2000 accepted of 4502 read (13 TeV), 5000 of 11291 (13.6 TeV).
FXFX_ACCEPTANCE = min(2000 / 4502, 5000 / 11291)


@pytest.mark.parametrize("workers", [1, 7, 28, 30])
@pytest.mark.parametrize("n_total", [200, 50000, 60000])
def test_chunks_use_each_event_once_without_vetoes(workers, n_total):
    plan = validate.chunk_plan(n_total, workers)
    assert sum(n for n, _ in plan) == n_total
    covered = [i for n, offset in plan for i in range(offset, offset + n)]
    assert sorted(covered) == list(range(n_total))


@pytest.mark.parametrize("workers", [1, 7, 28, 30])
@pytest.mark.parametrize("lhe_events", [50000, 150000])
def test_fxfx_chunks_never_read_into_each_other(workers, lhe_events):
    usable, lhe = validate.sample_budget({"events": lhe_events, "fxfx": {"qcut": 20.0}})
    assert lhe == lhe_events and usable <= FXFX_ACCEPTANCE * lhe_events
    plan = validate.chunk_plan(usable, workers, lhe)
    assert sum(n for n, _ in plan) == usable
    starts = [offset for _, offset in plan] + [lhe]
    for (n, offset), end in zip(plan, starts[1:]):
        # Each chunk reads n / acceptance LHE events on average; it must fit before the next chunk starts.
        assert offset + n / FXFX_ACCEPTANCE <= end


def test_unmerged_samples_are_used_whole():
    assert validate.sample_budget({"events": 60000}) == (60000, 60000)


# -- Stored validation results, recomputed independently ------------------------

def _recompute(plot: dict) -> tuple[float, float, float, int]:
    widths = [b - a for a, b in zip(plot["edges"][:-1], plot["edges"][1:])]
    rows = [(d, e, m, me, w) for d, e, m, me, w in zip(plot["data"], plot["data_err"], plot["mc"], plot["mc_err"], widths)
            if math.isfinite(d) and math.isfinite(m) and e * e + me * me > 0.0 and not (d == 0.0 and m == 0.0)]
    ndf = len(rows)
    chi2 = sum((m - d) ** 2 / (e * e + me * me) for d, e, m, me, _ in rows) / ndf
    ratio = sum(m * w for _, _, m, _, w in rows) / sum(d * w for d, _, _, _, w in rows)
    shape = sum((m / ratio - d) ** 2 / (e * e + (me / ratio) ** 2) for d, e, m, me, _ in rows) / ndf
    return chi2, shape, ratio, ndf


def _median(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return values[len(values) // 2]


@pytest.mark.parametrize("key", sorted(RESULTS))
def test_every_stored_chi2_follows_from_the_stored_histograms(key):
    entry = RESULTS[key]
    assert entry["plots"], key
    for plot in entry["plots"]:
        chi2, shape, ratio, ndf = _recompute(plot)
        assert plot["ndf"] == ndf
        assert plot["chi2_ndf"] == pytest.approx(chi2, rel=1e-9)
        assert plot["shape_chi2_ndf"] == pytest.approx(shape, rel=1e-9)
        assert plot["norm_ratio"] == pytest.approx(ratio, rel=1e-9)
    assert entry["median_chi2_ndf"] == pytest.approx(_median([p["chi2_ndf"] for p in entry["plots"]]))
    assert entry["median_shape_chi2_ndf"] == pytest.approx(_median([p["shape_chi2_ndf"] for p in entry["plots"]]))
    assert entry["n_plots"] == len(entry["plots"])


def test_results_cover_every_benchmark_and_variant_that_was_run():
    for bench in validate.BENCHMARKS:
        assert bench["key"] in RESULTS, bench["key"]
        for variant in bench.get("variants", []):
            key = f"{bench['key']}_{variant['suffix']}"
            assert key in RESULTS, key
            assert RESULTS[key]["variant"] == variant["suffix"]
            assert RESULTS[key]["generator"] == variant["label"]


def test_tuned_results_used_the_stored_tune():
    for key, entry in RESULTS.items():
        if entry["variant"].endswith("tuned"):
            assert entry["tune"] == TUNE["settings"], key


# -- Shower tune (hep/validation/tune.json) -------------------------------------

def test_stored_tune_is_reproduced_by_its_own_scan():
    settings, method, _fit, profile = tune.choose(TUNE["scan"])
    assert settings == TUNE["settings"] and method == TUNE["method"]
    assert profile["minimum"] == pytest.approx(TUNE["profile"]["minimum"])


def test_tune_lies_inside_pythia_limits_and_beats_monash():
    kt, pt0 = TUNE["settings"][tune.KT], TUNE["settings"][tune.PT0]
    assert 0.0 <= kt <= 10.0          # BeamRemnants:primordialKThard, PYTHIA 8.312 limits
    assert 0.5 <= pt0 <= 10.0         # SpaceShower:pT0Ref
    grid_kts = sorted({row["settings"][tune.KT] for row in TUNE["scan"]})
    assert grid_kts[0] < kt < grid_kts[-1]  # fitted inside the scanned range, not extrapolated
    assert TUNE["best_grid"]["chi2"] < TUNE["monash"]["chi2"]
    assert all(row["bins"] == TUNE["monash"]["bins"] for row in TUNE["scan"])


# -- Inputs agree with PDG 2026 (smlab/constants.py) -----------------------------

def _pdg_overrides() -> dict[str, float]:
    tree = ast.parse(_read("hep", "worker.py"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PDG_OVERRIDES")
    out = {}
    for line in ast.literal_eval(node.value):
        key, value = (part.strip() for part in line.split("="))
        out[key] = value if value in ("on", "off") else float(value)
    return out


def test_pythia_masses_and_widths_are_the_pdg_values():
    pdg = _pdg_overrides()
    assert pdg["23:m0"] == C.M_Z and pdg["23:mWidth"] == C.GAMMA_Z
    assert pdg["24:m0"] == C.M_W and pdg["24:mWidth"] == C.GAMMA_W
    assert pdg["25:m0"] == C.M_H and pdg["25:mWidth"] == pytest.approx(C.GAMMA_H)
    assert pdg["6:m0"] == C.M_T and pdg["6:mWidth"] == C.GAMMA_T
    assert pdg["15:m0"] == C.M_TAU
    assert pdg["StandardModel:sin2thetaWbar"] == C.SIN2_THETA_W
    # PYTHIA would otherwise recompute these widths from its partial widths.
    assert all(pdg[f"{pid}:doForceWidth"] == "on" for pid in (6, 23, 24, 25))


def _mg5_inputs() -> dict[str, float]:
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"^set (\w+) ([0-9.eE+-]+)$", _read("hep", "mg5", "generate_nlo.sh"), re.M)}


def test_madgraph_inputs_are_the_pdg_values():
    mg5 = _mg5_inputs()
    assert mg5["mt"] == C.M_T and mg5["ymt"] == C.M_T and mg5["wt"] == C.GAMMA_T
    assert mg5["mz"] == C.M_Z and mg5["wz"] == C.GAMMA_Z and mg5["ww"] == C.GAMMA_W
    assert mg5["gf"] == C.G_F
    assert mg5["lhaid"] == 303400  # NNPDF3.1 NLO, α_s(M_Z) = 0.118


def test_alpha_choice_reproduces_the_w_mass_in_the_gf_scheme():
    # Tree level: M_W² = M_Z²/2 · (1 + √(1 − 4πα / (√2 G_F M_Z²))).
    mg5 = _mg5_inputs()
    alpha = 1.0 / mg5["aewm1"]
    mz, gf = mg5["mz"], mg5["gf"]
    m_w = math.sqrt(mz * mz / 2.0 * (1.0 + math.sqrt(1.0 - 4.0 * math.pi * alpha / (math.sqrt(2.0) * gf * mz * mz))))
    assert m_w == pytest.approx(C.M_W, abs=0.002)  # within 2 MeV of the PDG average (±7.7 MeV)
    assert f"{m_w:.4f}" in _read("AUDIT_2026.md")  # the value quoted in the audit


# -- Numbers quoted in the documentation match the stored results ---------------

def _plot(key: str, suffix: str) -> dict:
    return next(p for p in RESULTS[key]["plots"] if p["path"].endswith(suffix))


def _claims() -> list[tuple[str, float, str]]:
    """(description, value from results/tune files, number as quoted in the docs)."""
    median = lambda key: RESULTS[key]["median_chi2_ndf"]  # noqa: E731
    return [
        ("Z pT LO", median("lhc_z_pt"), "4.23"),
        ("Z pT MC@NLO", median("lhc_z_pt_nlo"), "25.8"),
        ("Z pT FxFx", median("lhc_z_pt_fxfx"), "6.38"),
        ("Z pT FxFx + tune", median("lhc_z_pt_fxfx_tuned"), "1.65"),
        ("φ* LO", _plot("lhc_z_pt", "d28-x01-y01")["chi2_ndf"], "3.15"),
        ("φ* MC@NLO", _plot("lhc_z_pt_nlo", "d28-x01-y01")["chi2_ndf"], "15.9"),
        ("φ* FxFx", _plot("lhc_z_pt_fxfx", "d28-x01-y01")["chi2_ndf"], "2.91"),
        ("φ* FxFx + tune", _plot("lhc_z_pt_fxfx_tuned", "d28-x01-y01")["chi2_ndf"], "1.57"),
        ("t t̄ LO", median("lhc_ttbar"), "7.92"),
        ("t t̄ NLO", median("lhc_ttbar_nlo"), "2.51"),
        ("t t̄ MadSpin", median("lhc_ttbar_nlo_ms"), "2.01"),
        ("t t̄ MadSpin + tune", median("lhc_ttbar_ms_tuned"), "1.82"),
        ("eμ LO", median("lhc_ttbar_dilep"), "11.34"),
        ("eμ NLO", median("lhc_ttbar_dilep_nlo"), "1.50"),
        ("eμ MadSpin", median("lhc_ttbar_dilep_nlo_ms"), "0.97"),
        ("eμ MadSpin + tune", median("lhc_ttbar_dilep_ms_tuned"), "1.19"),
        ("LEP event shapes", median("lep_z_hadrons"), "2.48"),
        ("jets", median("lhc_jets"), "5.68"),
        ("merged FxFx σ [nb]", RESULTS["lhc_z_pt_fxfx"]["sigma_pb"] / 1000.0, "4.20"),
        ("tune kT [GeV]", TUNE["settings"][tune.KT], "3.04"),
        ("tune kT 1σ [GeV]", TUNE["profile"]["sigma"], "0.17"),
        ("Monash χ² in the fit region", TUNE["monash"]["chi2"], "259"),
    ]


@pytest.mark.parametrize("description,value,quoted", _claims(), ids=[c[0] for c in _claims()])
def test_quoted_numbers_match_the_results(description, value, quoted):
    decimals = len(quoted.split(".")[1]) if "." in quoted else 0
    assert f"{value:.{decimals}f}" == quoted, f"{description}: results give {value}"
    assert quoted in _read("AUDIT_2026.md"), f"{description}: {quoted} not quoted in AUDIT_2026.md"


METHODS_CLAIMS = [  # numbers quoted in the in-app Methods text
    ("lhc_ttbar", "7.92"), ("lhc_ttbar_nlo", "2.51"), ("lhc_ttbar_nlo_ms", "2.01"), ("lhc_ttbar_ms_tuned", "1.82"),
    ("lhc_ttbar_dilep_nlo_ms", "0.97"), ("lhc_ttbar_dilep_ms_tuned", "1.19"), ("lhc_z_pt_fxfx_tuned", "1.65"),
    ("lhc_z_pt_nlo", "25.8"),
]


@pytest.mark.parametrize("key,quoted", METHODS_CLAIMS)
def test_methods_text_quotes_the_stored_results(key, quoted):
    from smlab.methods import METHODS

    decimals = len(quoted.split(".")[1])
    assert f"{RESULTS[key]['median_chi2_ndf']:.{decimals}f}" == quoted
    assert quoted in METHODS


# -- The executable bundles everything the WSL side needs -----------------------

def test_every_bundled_file_exists():
    script = _read("build_exe.ps1")
    sources = re.findall(r'"([^";]+);[^"]+"', script)
    assert "hep\\worker.py" in sources and "hep\\ext\\smlab_fxfx.cpp" in sources
    for source in sources:
        assert os.path.exists(os.path.join(ROOT, source)), source
    # Scripts that the bundled scripts call must be bundled as well.
    for called in ("hep\\mg5\\decay_madspin.sh", "hep\\mg5\\sample_info.py", "hep\\ext\\build_fxfx.sh",
                   "hep\\mg5\\setup_mg5.sh", "hep\\mg5\\fix_dollar_formats.py", "hep\\mg5\\restore_weights.py"):
        assert called in sources, called


# -- Round 7: the comparison with other generators (hep/compare/) --------------------

COMPARE = os.path.join(HEP, "compare")
sys.path.insert(0, COMPARE)


def _round7_claims() -> list[tuple[str, float, str]]:
    import summary

    mg = summary.madgraph_table()
    table = dict(summary.generator_table())
    direct = summary.load("pythia_direct.json")
    claims = [("MadGraph max deviation [%]", 100 * mg["max_matched_deviation"], "0.13"),
              ("MadGraph points", mg["points"], "25")]
    for key, h, s_, d in (("lep_z_hadrons", "19.48", "4.57", "2.47"), ("lhc_z_pt", "4.62", "9.45", "4.09"),
                          ("lhc_ttbar", "15.59", "15.44", "8.67"), ("lhc_jets", "8.55", "6.27", "6.33")):
        claims += [(f"Herwig {key}", table[key]["herwig"], h), (f"Sherpa {key}", table[key]["sherpa"], s_),
                   (f"PYTHIA direct {key}", table[key]["pythia_direct"], d)]
    claims += [("Herwig minbias", table["lhc_minbias"]["herwig"], "7.10"),
               ("Herwig dilepton", table["lhc_ttbar_dilep"]["herwig"], "22.19"),
               ("Sherpa dilepton", table["lhc_ttbar_dilep"]["sherpa"], "15.53"),
               ("PYTHIA direct minbias", table["lhc_minbias"]["pythia_direct"], "36.7")]
    for key, quoted in (("lep_z_hadrons", "1.02"), ("lhc_minbias", "1.74"), ("lhc_z_pt", "1.00"),
                        ("lhc_ttbar", "0.70"), ("lhc_jets", "1.01")):
        claims.append((f"SMLab vs direct {key}", direct[key]["median_smlab_vs_direct"], quoted))
    return claims


@pytest.mark.skipif(not os.path.exists(os.path.join(COMPARE, "herwig.json")), reason="comparison not run")
@pytest.mark.parametrize("description,value,quoted", _round7_claims(), ids=[c[0] for c in _round7_claims()])
def test_round7_numbers_match_the_comparison_outputs(description, value, quoted):
    decimals = len(quoted.split(".")[1]) if "." in quoted else 0
    assert f"{value:.{decimals}f}" == quoted, f"{description}: outputs give {value}"
    assert quoted in _read("AUDIT_2026.md")


def test_builtin_engine_agrees_with_madgraph_in_the_same_scheme():
    rows = json.load(open(os.path.join(COMPARE, "builtin_vs_madgraph.json"), encoding="utf-8"))["rows"]
    for row in rows:
        # Statistical precision of MadGraph's integration plus 0.1 % for massive-fermion and width conventions.
        tolerance = 3 * row["madgraph_err_pb"] / row["madgraph_pb"] + 1e-3
        assert abs(row["matched_ratio"] - 1.0) < tolerance, (row["process"], row["sqrt_s"], row["matched_ratio"])
        if "madgraph_afb" in row:
            assert abs(row["smlab_matched_afb"] - row["madgraph_afb"]) < 3 * row["madgraph_afb_err"]
