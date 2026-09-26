"""Summarize Hadronica's comparison with other generators (Round 7 of AUDIT_2026.md).

    python hep/compare/summary.py

Reads the saved outputs of builtin_vs_madgraph.py, pythia_direct.py, and
other_generators.py (sherpa, herwig) plus Hadronica's own hep/validation/results.json,
and prints the tables quoted in AUDIT_2026.md.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def median(values) -> float:
    """The median as hep/validate.py defines it (upper middle element), so every column is comparable."""
    values = sorted(values)
    return values[len(values) // 2]


def load(name: str) -> dict:
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        return json.load(handle)


def madgraph_table() -> dict:
    rows = load("builtin_vs_madgraph.json")["rows"]
    matched = [abs(r["matched_ratio"] - 1.0) for r in rows]
    afb = [(r["hadronica_matched_afb"] - r["madgraph_afb"]) / r["madgraph_afb_err"] for r in rows if "madgraph_afb" in r]
    return {"points": len(rows), "processes": len({r["process"] for r in rows}),
            "max_matched_deviation": max(matched), "afb_pulls": afb}


def generator_table() -> list[tuple[str, dict]]:
    hadronica = json.load(open(os.path.join(ROOT, "hep", "validation", "results.json"), encoding="utf-8"))["benchmarks"]
    direct = load("pythia_direct.json")
    sherpa, herwig = load("sherpa.json"), load("herwig.json")
    best_variant = {"lhc_z_pt": "lhc_z_pt_fxfx_tuned", "lhc_ttbar": "lhc_ttbar_ms_tuned",
                    "lhc_ttbar_dilep": "lhc_ttbar_dilep_nlo_ms"}
    table = []
    for key in ("lep_z_hadrons", "lhc_minbias", "lhc_z_pt", "lhc_ttbar", "lhc_ttbar_dilep", "lhc_jets"):
        best_key = best_variant.get(key, key)
        direct_plots = direct.get(key, {}).get("plots", [])
        table.append((key, {
            "hadronica_lo": hadronica[key]["median_chi2_ndf"],
            "hadronica_best": hadronica[best_key]["median_chi2_ndf"],
            "hadronica_best_variant": hadronica[best_key]["variant"],
            "pythia_direct": median(p["direct_vs_data"] for p in direct_plots) if direct_plots else None,
            "sherpa": sherpa[key]["median_chi2_ndf"],
            "herwig": herwig[key]["median_chi2_ndf"],
        }))
    return table


def main() -> None:
    mg = madgraph_table()
    print(f"Built-in engine vs MadGraph5_aMC@NLO (LO, same scheme): {mg['processes']} processes, "
          f"{mg['points']} points, largest deviation {100 * mg['max_matched_deviation']:.2f} %; "
          f"A_FB pulls {', '.join(f'{p:+.1f}' for p in mg['afb_pulls'])}")
    direct = load("pythia_direct.json")
    print("Hadronica's PYTHIA mode vs PYTHIA run directly (MC-vs-MC median χ²/ndf):",
          ", ".join(f"{k} {v['median_hadronica_vs_direct']:.2f}" for k, v in direct.items()))
    print(f"{'benchmark':17s} {'Hadronica LO':>9s} {'PYTHIA':>7s} {'Herwig':>7s} {'Sherpa':>7s}   Hadronica best")
    for key, row in generator_table():
        fmt = lambda v: "   —   " if v is None else f"{v:7.2f}"  # noqa: E731
        print(f"{key:17s} {row['hadronica_lo']:9.2f} {fmt(row['pythia_direct'])} {fmt(row['herwig'])} "
              f"{fmt(row['sherpa'])}   {row['hadronica_best']:.2f} ({row['hadronica_best_variant']})")


if __name__ == "__main__":
    main()
