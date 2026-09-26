"""Re-tune PYTHIA's non-perturbative initial-state parameters for Hadronica's NLO samples.

    ~/micromamba/envs/hadronica-hep/bin/python hep/tune.py [--events N] [--quick]

Method (after the ATLAS AZNLO tune, JHEP 09 (2014) 145): with the hard
process and the first emission fixed at NLO (here the FxFx-merged Z + 0, 1, 2
jet sample), the shape of the Z-boson transverse momentum below ~30 GeV is set
by the Sudakov region of the initial-state shower and by the intrinsic
transverse momentum of the partons in the proton. Two parameters control it:

    BeamRemnants:primordialKThard   Gaussian width of the intrinsic kT (GeV)
    SpaceShower:pT0Ref              ISR regularization scale at 7 TeV (GeV)

The shower α_s stays at 0.118, as MC@NLO/FxFx matching requires. Each grid
point showers the same LHE events (so MC fluctuations largely cancel between
points), is analysed with Rivet (ATLAS_2019_I1768911, Eur. Phys. J. C 80 (2020)
616), and is compared with the measured shape (MC normalized to the data in
the fit region) of p_T^ll < 30 GeV and φ*_η < 0.3. A quadratic
χ²(k_T, p_T0) surface is fitted to the grid and its minimum, if it lies inside
the grid, is the tune. Otherwise p_T0 is fixed at its best grid value (which
can be PYTHIA's lower limit of 0.5 GeV) and k_T is the minimum of a quadratic
fitted along that row; failing that, the best grid point is used.
``--refit tune.json`` repeats the fit on a saved scan. The result goes
to hep/validation/tune.json, which validate.py (the "*_tuned" variants) and
the app's NLO mode apply. Top-pair data are not used, so the t t̄ benchmarks
test the tune on independent data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import validate  # noqa: E402

SAMPLE = "dy_fxfx"
KT = "BeamRemnants:primordialKThard"
PT0 = "SpaceShower:pT0Ref"
GRID = {KT: [1.0, 1.4, 1.8, 2.2, 2.6], PT0: [0.8, 1.4, 2.0, 2.6]}
MONASH = {KT: 1.8, PT0: 2.0}
# Fit region: the Sudakov/non-perturbative part of each distribution.
TARGETS = {
    "/ATLAS_2019_I1768911/d27-x01-y01": 30.0,   # p_T^ll [GeV]
    "/ATLAS_2019_I1768911/d28-x01-y01": 0.3,    # φ*_η
}


def region_chi2(plots: list[dict]) -> tuple[float, int]:
    """Shape χ² of the target histograms in their fit regions: (sum, number of bins)."""
    total, bins = 0.0, 0
    for plot in plots:
        limit = TARGETS.get(plot["path"])
        if limit is None:
            continue
        rows = [
            (d, e, m, me, hi - lo)
            for lo, hi, d, e, m, me in zip(plot["edges"][:-1], plot["edges"][1:], plot["data"],
                                           plot["data_err"], plot["mc"], plot["mc_err"])
            if hi <= limit + 1e-9 and e > 0.0
        ]
        data_area = sum(d * w for d, _e, _m, _me, w in rows)
        mc_area = sum(m * w for _d, _e, m, _me, w in rows)
        if not rows or mc_area <= 0.0:
            continue
        r = mc_area / data_area
        total += sum((m / r - d) ** 2 / (e * e + (me / r) ** 2) for d, e, m, me, _w in rows)
        bins += len(rows)
    return total, bins


def fit_quadratic(points: list[tuple[float, float, float]]) -> dict | None:
    """Least-squares χ²(x, y) = c0 + c1 x + c2 y + c3 x² + c4 y² + c5 x y; its minimum if it is one."""
    import numpy as np

    x, y, z = (np.array(v) for v in zip(*points))
    design = np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    c0, c1, c2, c3, c4, c5 = coef
    hessian = np.array([[2 * c3, c5], [c5, 2 * c4]])
    if np.any(np.linalg.eigvalsh(hessian) <= 0.0):
        return {"coefficients": coef.tolist(), "minimum": None}
    xm, ym = np.linalg.solve(hessian, [-c1, -c2])
    zm = float(c0 + c1 * xm + c2 * ym + c3 * xm * xm + c4 * ym * ym + c5 * xm * ym)
    # 1σ: the Δχ² = 1 contour of the fitted surface, projected on each axis.
    cov = np.linalg.inv(hessian / 2.0)
    return {
        "coefficients": coef.tolist(),
        "minimum": [float(xm), float(ym)],
        "chi2_min": zm,
        "sigma": [float(math.sqrt(cov[0, 0])), float(math.sqrt(cov[1, 1]))],
    }


def fit_profile(points: list[tuple[float, float]]) -> dict | None:
    """Least-squares χ²(x) = a + b x + c x²; its minimum and Δχ² = 1 half-width if c > 0."""
    import numpy as np

    x, z = (np.array(v) for v in zip(*points))
    c, b, a = np.polyfit(x, z, 2)
    if c <= 0.0:
        return None
    xm = -b / (2.0 * c)
    return {"coefficients": [float(a), float(b), float(c)], "minimum": float(xm),
            "chi2_min": float(a + b * xm + c * xm * xm), "sigma": float(1.0 / math.sqrt(c))}


def choose(grid_points: list[dict]) -> tuple[dict, str, dict | None, dict | None]:
    """(settings, method, 2D fit, 1D profile fit) from the scanned grid points."""
    kts = [row["settings"][KT] for row in grid_points]
    pt0s = [row["settings"][PT0] for row in grid_points]
    best = min(grid_points, key=lambda row: row["chi2"])
    fit = fit_quadratic([(row["settings"][KT], row["settings"][PT0], row["chi2"]) for row in grid_points])
    if (fit and fit["minimum"] is not None and min(kts) <= fit["minimum"][0] <= max(kts)
            and min(pt0s) <= fit["minimum"][1] <= max(pt0s)):
        settings = {KT: round(fit["minimum"][0], 3), PT0: round(fit["minimum"][1], 3)}
        return settings, "minimum of a quadratic fit to the χ² grid", fit, None
    pt0 = best["settings"][PT0]
    row = sorted((r["settings"][KT], r["chi2"]) for r in grid_points if r["settings"][PT0] == pt0)
    profile = fit_profile(row) if len(row) >= 3 else None
    if profile and row[0][0] <= profile["minimum"] <= row[-1][0]:
        profile["fixed"] = {PT0: pt0}
        settings = {KT: round(profile["minimum"], 2), PT0: pt0}
        method = (f"{PT0} fixed at its best grid value {pt0} GeV; {KT} from a quadratic fit along that row "
                  "(the 2D quadratic has no minimum inside the grid)")
        return settings, method, fit, profile
    return dict(best["settings"]), "best grid point (no fitted minimum inside the grid)", fit, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=60000, help="events per grid point")
    parser.add_argument("--quick", action="store_true", help="2 × 2 grid, 2000 events, to test the pipeline")
    parser.add_argument("--kt", type=float, nargs="*", default=GRID[KT], help="primordialKThard grid (GeV)")
    parser.add_argument("--pt0", type=float, nargs="*", default=GRID[PT0], help="SpaceShower:pT0Ref grid (GeV)")
    parser.add_argument("--sample", default=SAMPLE, help="Z sample at 13 TeV (default: the FxFx-merged one)")
    parser.add_argument("--refit", default=None, help="refit the scan saved in this tune.json (no generation)")
    args = parser.parse_args()
    if args.refit:
        with open(args.refit, encoding="utf-8") as handle:
            tune = json.load(handle)
        settings, method, fit, profile = choose(tune["scan"])
        tune.update(settings=settings, method=method, fit=fit, profile=profile)
        with open(args.refit, "w", encoding="utf-8") as handle:
            json.dump(tune, handle, indent=1)
        print(f"tune: {settings} ({method}) -> {args.refit}")
        return
    os.environ["PATH"] = validate.BIN + os.pathsep + os.environ.get("PATH", "")
    from worker import nlo_samples

    sample = nlo_samples(all_variants=True).get(f"pp_z_ll@13000#{args.sample}")
    if sample is None:
        sys.exit(f"no {args.sample} sample at 13 TeV; run hep/mg5/produce_all.sh first")
    bench = dict(next(b for b in validate.BENCHMARKS if b["key"] == "lhc_z_pt"))
    grid = {KT: args.kt, PT0: args.pt0}
    if args.quick:
        grid = {k: v[1:3] for k, v in grid.items()}
    events = 2000 if args.quick else args.events
    bench["events"] = events
    started = time.time()
    scan = []
    with tempfile.TemporaryDirectory(prefix="hadronica-tune-") as work:
        validate.OUT_DIR = work
        points = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
        for settings in [dict(MONASH)] + points:
            source = {"source": "nlo", "sample": args.sample, "tune": settings}
            available, lhe_events = validate.sample_budget(sample)
            result = validate.run(bench, 1.0, source, "tune", available, "tune", lhe_events)
            chi2, bins = region_chi2(result["plots"])
            scan.append({"settings": settings, "chi2": chi2, "bins": bins})
            print(f"kT {settings[KT]:.2f}  pT0 {settings[PT0]:.2f}  χ²/bins {chi2:.1f}/{bins} "
                  f"= {chi2 / max(bins, 1):.2f}  ({result['seconds']:.0f}s)", flush=True)
    monash = scan[0]
    grid_points = scan[1:]
    best = min(grid_points, key=lambda row: row["chi2"])
    settings, method, fit, profile = choose(grid_points)
    tune = {
        "name": "Hadronica-AZ 2026",
        "settings": settings,
        "method": method,
        "sample": f"{args.sample}_13000 ("
                  + ("MadGraph5_aMC@NLO FxFx Z+0,1,2j" if args.sample == SAMPLE else "MadGraph5_aMC@NLO") + ")",
        "data": "ATLAS_2019_I1768911 (Eur. Phys. J. C 80 (2020) 616): p_T^ll < 30 GeV, φ*_η < 0.3, shape only",
        "reference_method": "ATLAS AZNLO, JHEP 09 (2014) 145",
        "events_per_point": events,
        "monash": {"settings": monash["settings"], "chi2": monash["chi2"], "bins": monash["bins"]},
        "best_grid": best,
        "fit": fit,
        "profile": profile,
        "scan": grid_points,
        "seconds": round(time.time() - started),
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    path = os.path.join(HERE, "validation", "tune-quick.json" if args.quick else "tune.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(tune, handle, indent=1)
    print(f"tune: {settings} ({method}); Monash χ² {monash['chi2']:.1f}, best grid {best['chi2']:.1f}"
          f" over {best['bins']} bins -> {path}")


if __name__ == "__main__":
    main()
