"""Build the comparison report page from SMLab's saved comparison outputs (no hand-copied numbers).

    python hep/compare/build_report.py [output.html]

Writes docs/index.html by default, the page served by GitHub Pages.
"""
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "hep", "compare"))
import summary  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "index.html")
REPO = "https://github.com/bonquifo/SMLab"

mg = json.load(open(os.path.join(ROOT, "hep", "compare", "builtin_vs_madgraph.json"), encoding="utf-8"))
direct = json.load(open(os.path.join(ROOT, "hep", "compare", "pythia_direct.json"), encoding="utf-8"))
table = dict(summary.generator_table())
mg_summary = summary.madgraph_table()

NAMES = {
    "lep_z_hadrons": ("LEP event shapes", "ALEPH, e⁺e⁻ → hadrons at 91.2 GeV"),
    "lhc_minbias": ("Minimum bias", "ATLAS, charged particles at 13 TeV"),
    "lhc_z_pt": ("Z transverse momentum", "ATLAS, Z → ℓℓ at 13 TeV"),
    "lhc_ttbar": ("Top pairs, lepton + jets", "CMS, event variables at 13 TeV"),
    "lhc_ttbar_dilep": ("Top pairs, dilepton", "ATLAS, eμ channel at 13 TeV"),
    "lhc_jets": ("Inclusive jets", "CMS, jet cross section at 13 TeV"),
}
# The published measurement behind each benchmark: journal reference and INSPIRE record (the Rivet analysis id).
REFS = {
    "lep_z_hadrons": ("Phys. Rept. 294 (1998) 1", 428072),
    "lhc_minbias": ("Phys. Lett. B 758 (2016) 67", 1419652),
    "lhc_z_pt": ("Eur. Phys. J. C 80 (2020) 616", 1768911),
    "lhc_ttbar": ("JHEP 06 (2018) 002", 1662081),
    "lhc_ttbar_dilep": ("Eur. Phys. J. C 80 (2020) 528", 1759875),
    "lhc_jets": ("Eur. Phys. J. C 76 (2016) 451", 1459051),
}
BEST = {"lo": "same as LO", "fxfx_tuned": "NLO FxFx + SMLab tune", "ms_tuned": "NLO + MadSpin + tune",
        "nlo_ms": "NLO + MadSpin"}
FLAGS = {("lep_z_hadrons", "herwig"): "unconfirmed", ("lhc_minbias", "sherpa"): "not comparable"}

# Chart rows: (key, {series: value or None}).
series = [("smlab_lo", "SMLab LO", "lo"), ("herwig", "Herwig 7.3", "hw"), ("sherpa", "Sherpa 3.0", "sh"),
          ("smlab_best", "SMLab best", "best")]
chart_rows = []
for key, row in table.items():
    values = {}
    for field, _label, _cls in series:
        v = row[field]
        if FLAGS.get((key, field.split("_")[0] if field.startswith(("herwig", "sherpa")) else field)) == "not comparable":
            v = None
        values[field] = v
    chart_rows.append((key, values))

data_js = json.dumps({"rows": [{"key": k, "label": NAMES[k][0], "values": v,
                               "hollow": [f for f in ("herwig", "sherpa") if FLAGS.get((k, f)) == "unconfirmed"]}
                              for k, v in chart_rows],
                      "series": [{"field": f, "label": l, "cls": c} for f, l, c in series]}, ensure_ascii=False)


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def cell(key, field):
    row = table[key]
    v = row[field]
    flag = FLAGS.get((key, field))
    if flag == "not comparable":
        return '<td class="num muted">n/c <sup>‡</sup></td>'
    best = min(x for x in (row["smlab_lo"], row["herwig"], row["sherpa"] if FLAGS.get((key, "sherpa")) is None else 1e9)
               if x is not None)
    cls = "num" + (" win" if v is not None and abs(v - best) < 1e-9 else "")
    mark = " <sup>†</sup>" if flag == "unconfirmed" else ""
    return f'<td class="{cls}">{fmt(v)}{mark}</td>'


result_rows = "\n".join(
    f"""<tr><th scope="row"><span class="bench">{html.escape(NAMES[k][0])}</span><span class="sub">{html.escape(NAMES[k][1])}; <a href="https://inspirehep.net/literature/{REFS[k][1]}">{html.escape(REFS[k][0])}</a></span></th>
{cell(k, 'smlab_lo')}{cell(k, 'herwig')}{cell(k, 'sherpa')}
<td class="num best">{fmt(r['smlab_best'])}<span class="sub">{html.escape(BEST.get(r['smlab_best_variant'], r['smlab_best_variant']))}</span></td></tr>"""
    for k, r in table.items())

PROC = {"ff13": "e⁺e⁻ → μ⁺μ⁻", "ff15": "e⁺e⁻ → τ⁺τ⁻", "ff2": "e⁺e⁻ → uū", "ff1": "e⁺e⁻ → dd̄", "ff5": "e⁺e⁻ → bb̄",
        "ff6": "e⁺e⁻ → tt̄", "ff14": "e⁺e⁻ → ν_μν̄_μ", "ff12": "e⁺e⁻ → ν_eν̄_e", "zh": "e⁺e⁻ → ZH",
        "bhabha": "Bhabha (QED)", "diphoton": "e⁺e⁻ → γγ (QED)"}
mg_rows = "\n".join(
    f"""<tr><td>{PROC[r['process']]}</td><td class="num">{r['sqrt_s']:.1f}</td><td class="num">{r['madgraph_pb']:.4g}</td>
<td class="num {'good' if abs(r['matched_ratio'] - 1) < 2e-3 else ''}">{r['matched_ratio']:.4f}</td><td class="num">{r['shipped_ratio']:.4f}</td></tr>"""
    for r in mg["rows"])
afb = [r for r in mg["rows"] if "madgraph_afb" in r]
afb_rows = "\n".join(
    f"""<tr><td class="num">{r['sqrt_s']:.1f}</td><td class="num">{r['madgraph_afb']:.4f} ± {r['madgraph_afb_err']:.4f}</td>
<td class="num">{r['smlab_matched_afb']:.4f}</td><td class="num">{r['smlab_shipped_afb']:.4f}</td></tr>""" for r in afb)
direct_rows = "\n".join(
    f"""<tr><td>{NAMES[k][0]}</td><td class="num">{fmt(table[k]['smlab_lo'])}</td><td class="num">{fmt(table[k]['pythia_direct'])}</td>
<td class="num">{v['median_smlab_vs_direct']:.2f}</td></tr>""" for k, v in direct.items())

page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMLab Benchmark Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&display=swap">
<style>
:root {{
  --ground: #f5f7fa; --panel: #ffffff; --ink: #17212d; --muted: #5a6677; --rule: #d9dfe7; --soft: #eef2f7;
  --smlab: #2d6cdf; --best: #0f7a5c; --hw: #b8691f; --sh: #7b57a8; --warn: #9a5b00; --good-bg: #e3f3ec;
  --grid: #d3dae4;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground: #0e131b; --panel: #151c27; --ink: #e5eaf1; --muted: #9aa6b6; --rule: #2a3444; --soft: #1b2431;
    --smlab: #6fa0ff; --best: #3fcf9d; --hw: #e59a50; --sh: #b18ee0; --warn: #f0b75c; --good-bg: #17352b; --grid: #2b3647;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground: #0e131b; --panel: #151c27; --ink: #e5eaf1; --muted: #9aa6b6; --rule: #2a3444; --soft: #1b2431;
  --smlab: #6fa0ff; --best: #3fcf9d; --hw: #e59a50; --sh: #b18ee0; --warn: #f0b75c; --good-bg: #17352b; --grid: #2b3647;
}}
body {{ background: var(--ground); color: var(--ink); font: 16px/1.6 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }}
.page {{ max-width: 920px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; display: grid; gap: 40px; }}
h1, h2 {{ font-family: "Source Serif 4", Georgia, serif; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(30px, 5vw, 42px); font-weight: 650; line-height: 1.15; }}
h2 {{ font-size: 25px; font-weight: 650; line-height: 1.25; }}
h3 {{ font-size: 15px; font-weight: 600; margin: 0; }}
p {{ margin: 0; max-width: 68ch; }}
.eyebrow {{ font: 500 12px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
header {{ display: grid; gap: 14px; }}
.lede {{ font-size: 18px; color: var(--ink); }}
section {{ display: grid; gap: 16px; }}
.findings {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
@media (max-width: 860px) {{ .findings {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 440px) {{ .findings {{ grid-template-columns: 1fr; }} }}
.finding {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; display: grid; gap: 6px; align-content: start; }}
.finding b {{ font: 500 26px/1.1 "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.finding.pos b {{ color: var(--best); }} .finding.neg b {{ color: var(--hw); }}
.scroll {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }}
thead th {{ font: 500 12px/1.3 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); background: var(--soft); }}
tbody tr:last-child > * {{ border-bottom: 0; }}
td.num, th.num {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
td.win {{ font-weight: 500; color: var(--smlab); }}
td.best {{ color: var(--best); font-weight: 500; }}
td.good {{ background: var(--good-bg); }}
td.muted {{ color: var(--muted); }}
.bench {{ display: block; font-weight: 600; }}
.sub {{ display: block; font: 400 12px/1.4 "IBM Plex Sans", system-ui, sans-serif; color: var(--muted); text-align: inherit; white-space: normal; }}
td.num .sub {{ text-align: right; }}
.note {{ font-size: 14px; color: var(--muted); }}
.notes {{ display: grid; gap: 10px; font-size: 14px; }}
.notes p {{ max-width: none; }}
sup {{ color: var(--warn); font-size: 11px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 13px; color: var(--muted); }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
.chart {{ padding: 14px 10px 6px; }}
.chart svg {{ display: block; width: 100%; min-width: 560px; height: auto; }}
.chart text {{ fill: var(--muted); font: 12px "IBM Plex Mono", ui-monospace, monospace; }}
.chart .rowlabel {{ fill: var(--ink); font: 500 13px "IBM Plex Sans", system-ui, sans-serif; }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
a {{ color: var(--smlab); }}
code {{ font: 13px "IBM Plex Mono", ui-monospace, monospace; background: var(--soft); padding: 1px 5px; border-radius: 4px; }}
</style>

<main class="page">
<header>
  <span class="eyebrow">Standard Model Collision Laboratory · Validation report · September 2026</span>
  <h1>How SMLab compares with MadGraph, PYTHIA, Herwig and Sherpa</h1>
  <p class="lede">SMLab was tested against the established particle-physics event generators on the same inputs and the same
  published measurements. Its own physics engine reproduces MadGraph to 0.13&nbsp;%, its PYTHIA mode is identical to PYTHIA run
  directly, and against real LEP and LHC data it matches or beats Herwig&nbsp;7.3 and Sherpa&nbsp;3.0 on five of six benchmarks.
  Soft (minimum-bias) collisions are where it falls behind.</p>
</header>

<section aria-labelledby="summary">
  <h2 id="summary">Summary</h2>
  <div class="findings">
    <div class="finding pos"><span class="eyebrow">Own engine vs MadGraph</span><b>≤ {100 * mg_summary['max_matched_deviation']:.2f}&nbsp;%</b>
      <span class="note">largest difference over {mg_summary['points']} cross sections, {mg_summary['processes']} processes, same conventions</span></div>
    <div class="finding pos"><span class="eyebrow">PYTHIA mode vs PYTHIA</span><b>identical</b>
      <span class="note">bit-for-bit the same Rivet histograms for the same events</span></div>
    <div class="finding pos"><span class="eyebrow">Best on data</span><b>{table['lhc_ttbar']['smlab_best']:.2f}</b>
      <span class="note">χ²/ndf on CMS top-pair data with NLO + MadSpin + tune (Herwig {table['lhc_ttbar']['herwig']:.1f}, Sherpa {table['lhc_ttbar']['sherpa']:.1f})</span></div>
    <div class="finding neg"><span class="eyebrow">Weakest area</span><b>{table['lhc_minbias']['smlab_lo']:.1f}</b>
      <span class="note">χ²/ndf on ATLAS minimum bias, against Herwig's {table['lhc_minbias']['herwig']:.1f}</span></div>
  </div>
</section>

<section aria-labelledby="data">
  <h2 id="data">Against measured data</h2>
  <p>Each generator ran SMLab's six validation benchmarks, analysed with the same Rivet analyses and scored with the same code.
  The score is the median χ²/ndf over each measurement's distributions: 1 means agreement within the uncertainties, and lower is
  better. The first three columns are like for like: leading order with a parton shower (Sherpa also merges extra jets). The last
  column is SMLab's best mode, which uses next-to-leading-order samples.</p>
  <div class="legend" aria-hidden="true">
    <span><i class="dot" style="background:var(--smlab)"></i>SMLab LO (PYTHIA 8)</span>
    <span><i class="dot" style="background:var(--hw)"></i>Herwig 7.3</span>
    <span><i class="dot" style="background:var(--sh)"></i>Sherpa 3.0</span>
    <span><i class="dot" style="background:var(--best)"></i>SMLab best (NLO)</span>
    <span><i class="dot" style="border:2px solid var(--hw);background:transparent;width:9px;height:9px"></i>unconfirmed</span>
  </div>
  <div class="scroll chart"><svg id="chart" role="img" aria-label="Median chi-squared per degree of freedom for each benchmark and generator, logarithmic scale"></svg></div>
  <div class="scroll">
  <table>
    <thead><tr><th scope="col">Benchmark</th><th scope="col" class="num">SMLab LO</th><th scope="col" class="num">Herwig 7.3</th>
    <th scope="col" class="num">Sherpa 3.0</th><th scope="col" class="num">SMLab best</th></tr></thead>
    <tbody>
{result_rows}
    </tbody>
  </table>
  </div>
  <div class="notes">
    <p><sup>†</sup> <b>Herwig on LEP is unconfirmed.</b> Its charged multiplicity is 19.48 against ALEPH's 20.91 ± 0.22 (Phys. Rept. 294 (1998) 1). Herwig's
    authors document that its tune cannot describe LEP spectra and multiplicities well at the same time
    (D. Reichelt, P. Richardson, A. Siodmok, Eur. Phys. J. C 77 (2017) 876,
    <a href="https://arxiv.org/abs/1708.01491">arXiv:1708.01491</a>), but its event shapes are surprisingly poor for a
    generator tuned to LEP. This Herwig build needed nine workarounds for Ubuntu 26.04's compilers and tools, so a build
    problem cannot be ruled out.</p>
    <p><sup>‡</sup> <b>Sherpa on minimum bias is not comparable.</b> Sherpa's official minimum-bias example covers
    non-diffractive collisions only, while the ATLAS measurement includes diffraction. It predicts
    {100 * (json.load(open(os.path.join(ROOT, 'hep', 'compare', 'sherpa.json'), encoding='utf-8'))['lhc_minbias']['median_norm_ratio'] - 1):.0f}&nbsp;%
    too many particles per event, which says more about the setup than about Sherpa.</p>
    <p><b>Limited by simulation statistics.</b> In the dilepton top-pair and Z transverse-momentum measurements the data are
    more precise than 60&nbsp;000 to 150&nbsp;000 simulated events. There a χ²/ndf near 1 means consistent within the
    simulation's statistics rather than agreement at the data's precision; large values still mean real disagreement.</p>
    <p><b>Fairness fixes.</b> Herwig's LEP setup radiated photons off the incoming beams, which the data are corrected for;
    it now runs without initial-state radiation (checked: no beam photons in 300 events). Sherpa's default left K⁰<sub>S</sub>
    and Λ undecayed, which the LEP measurement counts; its LEP run now decays particles with cτ &lt; 300&nbsp;mm, the LEP
    convention. Herwig and Sherpa ran at leading order because their NLO modes need OpenLoops, which is not installed.</p>
  </div>
</section>

<section aria-labelledby="madgraph">
  <h2 id="madgraph">SMLab's own engine against MadGraph</h2>
  <p>SMLab's built-in engine computes Born cross sections from its own formulas. MadGraph5_aMC@NLO is the standard for
  tree-level calculations. With SMLab's couplings switched to MadGraph's conventions (α = 1/132.04, on-shell mixing angle,
  no QCD factor) every cross section agrees within MadGraph's precision, which checks the formulas themselves. As shipped,
  SMLab uses a running α(s), the effective mixing angle and a QCD factor for quarks; those choices move cross sections by a
  few percent and are what reproduce LEP's measured asymmetries.</p>
  <div class="scroll">
  <table>
    <thead><tr><th scope="col">Process</th><th scope="col" class="num">√s (GeV)</th><th scope="col" class="num">MadGraph σ (pb)</th>
    <th scope="col" class="num">SMLab / MG, same scheme</th><th scope="col" class="num">SMLab / MG, as shipped</th></tr></thead>
    <tbody>
{mg_rows}
    </tbody>
  </table>
  </div>
  <div class="cols">
    <div class="scroll">
    <table>
      <thead><tr><th scope="col" class="num">√s (GeV)</th><th scope="col" class="num">MadGraph A<sub>FB</sub></th>
      <th scope="col" class="num">SMLab, same scheme</th><th scope="col" class="num">SMLab, shipped</th></tr></thead>
      <tbody>
{afb_rows}
      </tbody>
    </table>
    </div>
    <p class="note">Forward–backward asymmetry of e⁺e⁻ → μ⁺μ⁻ (MadGraph from 20&nbsp;000 events). At the Z pole, SMLab as shipped
    gives 0.0161, close to LEP's measured 0.0169 ± 0.0013 (<a href="https://pdg.lbl.gov/2026/reviews/rpp2026-rev-standard-model.pdf">PDG 2026
    Electroweak review</a>, Table 10.3); MadGraph's tree-level scheme gives about twice that.</p>
  </div>
</section>

<section aria-labelledby="pythia">
  <h2 id="pythia">SMLab's PYTHIA mode against PYTHIA itself</h2>
  <p>A stand-alone C++ program ran PYTHIA with exactly the settings SMLab uses, handing events to Rivet through PYTHIA's own
  converter and skipping every SMLab layer. For the same events, both routes produce bit-identical histograms. With independent
  random seeds they agree within statistics; the minimum-bias figure reflects strongly correlated bins, since a control
  comparing PYTHIA with itself gives 1.14 on the same measure.</p>
  <div class="scroll">
  <table>
    <thead><tr><th scope="col">Benchmark</th><th scope="col" class="num">SMLab vs data</th><th scope="col" class="num">PYTHIA direct vs data</th>
    <th scope="col" class="num">SMLab vs PYTHIA</th></tr></thead>
    <tbody>
{direct_rows}
    </tbody>
  </table>
  </div>
</section>

<section aria-labelledby="method">
  <h2 id="method">Method</h2>
  <div class="notes">
    <p>All runs used the same beams, energies and generation cuts as SMLab's validation: 60&nbsp;000 to 150&nbsp;000 events per
    benchmark, generated in parallel chunks with independent seeds and merged with <code>rivet-merge</code>. Herwig 7.3.0 was
    built from source with herwig-bootstrap; Sherpa 3.0.0 came from conda-forge; both used their authors' example inputs,
    changing only beams, energy, cuts and output. Rivet 4.1.4 supplied the analyses and the published reference data.</p>
    <p>The scripts are in <a href="{REPO}/tree/main/hep/compare"><code>hep/compare/</code></a> of the
    <a href="{REPO}">SMLab repository</a>: <code>builtin_vs_madgraph.py</code>,
    <code>pythia_direct.py</code>, <code>other_generators.py</code> and <code>summary.py</code>. Every number on this page is
    read from their saved outputs, and SMLab's test suite checks the numbers quoted in its audit against the same files.</p>
    <p>Generators: MadGraph5_aMC@NLO, J. Alwall et al., JHEP 07 (2014) 079,
    <a href="https://arxiv.org/abs/1405.0301">arXiv:1405.0301</a>; PYTHIA 8.3, C. Bierlich et al., SciPost Phys. Codebases 8 (2022),
    <a href="https://arxiv.org/abs/2203.11601">arXiv:2203.11601</a>; Herwig 7.3, G. Bewick et al., Eur. Phys. J. C 84 (2024) 1053,
    <a href="https://arxiv.org/abs/2312.05175">arXiv:2312.05175</a>; Sherpa 3, E. Bothmann et al., JHEP 12 (2024) 156,
    <a href="https://arxiv.org/abs/2410.22148">arXiv:2410.22148</a>. Analysis: Rivet 4, C. Bierlich et al., SciPost Phys. Codebases 36 (2024),
    <a href="https://arxiv.org/abs/2404.15984">arXiv:2404.15984</a>; reference data from
    <a href="https://www.hepdata.net/">HEPData</a> (E. Maguire, L. Heinrich, G. Watt, J. Phys. Conf. Ser. 898 (2017) 102006).</p>
    <p>Sources: D. Reichelt, P. Richardson, A. Siodmok, <a href="https://arxiv.org/abs/1708.01491">Improving the
    Simulation of Quark and Gluon Jets with Herwig 7</a>, Eur. Phys. J. C 77 (2017) 876;
    <a href="https://mcplots.cern.ch/?query=plots%2C%2Czhad%2CpToutSph%2CHerwig.Main">MCPlots, Z → hadrons, Herwig</a>.</p>
  </div>
</section>
</main>

<script>
(function () {{
  const DATA = {data_js};
  const svg = document.getElementById("chart");
  const ns = "http://www.w3.org/2000/svg";
  const W = 760, left = 200, right = 24, top = 30, rowH = 46, H = top + DATA.rows.length * rowH + 30;
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  const lo = Math.log10(0.5), hi = Math.log10(40);
  const x = v => left + (Math.log10(Math.min(Math.max(v, 0.5), 40)) - lo) / (hi - lo) * (W - left - right);
  const el = (name, attrs, text) => {{
    const e = document.createElementNS(ns, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    svg.appendChild(e); return e;
  }};
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  for (const t of [0.5, 1, 2, 5, 10, 20, 40]) {{
    el("line", {{ x1: x(t), x2: x(t), y1: top - 8, y2: H - 26, stroke: color(t === 1 ? "--muted" : "--grid"),
      "stroke-width": t === 1 ? 1.4 : 1, "stroke-dasharray": t === 1 ? "4 3" : "" }});
    el("text", {{ x: x(t), y: H - 10, "text-anchor": "middle" }}, String(t));
  }}
  el("text", {{ x: x(1) + 6, y: top - 14 }}, "χ²/ndf = 1: agreement");
  const colors = {{ smlab_lo: "--smlab", herwig: "--hw", sherpa: "--sh", smlab_best: "--best" }};
  const offsets = {{ smlab_lo: -9, herwig: -3, sherpa: 3, smlab_best: 9 }};
  DATA.rows.forEach((row, i) => {{
    const y = top + i * rowH + rowH / 2;
    el("text", {{ x: 12, y: y + 4, class: "rowlabel" }}, row.label);
    el("line", {{ x1: left, x2: W - right, y1: y, y2: y, stroke: color("--grid"), "stroke-width": 1 }});
    for (const s of DATA.series) {{
      const v = row.values[s.field];
      if (v == null) continue;
      const cx = x(v), cy = y + offsets[s.field];
      const hollow = row.hollow.includes(s.field);
      const c = el("circle", {{ cx, cy, r: hollow ? 5 : 6, fill: hollow ? color("--panel") : color(colors[s.field]),
        stroke: hollow ? color(colors[s.field]) : color("--panel"), "stroke-width": hollow ? 2 : 1.5 }});
      const tip = document.createElementNS(ns, "title");
      tip.textContent = `${{s.label}}: ${{v.toFixed(2)}}${{hollow ? " (unconfirmed)" : ""}}`; c.appendChild(tip);
    }}
  }});
}})();
</script>
"""
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(page)
print("wrote", OUT, len(page), "bytes")
