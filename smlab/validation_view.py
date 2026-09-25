"""The Validation panel: SMLab's generators against published measurements (Rivet).

Reads the results.json written by hep/validate.py and draws, for each
benchmark, the agreement summary and every compared distribution with the
data points, the generator predictions, and their ratio.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pygame

from smlab.theme import (
    ACCENT,
    ACCENT_SOFT,
    BAD,
    BG,
    BORDER,
    GOLD,
    GOOD,
    SURFACE_2,
    SURFACE_3,
    TEXT,
    TEXT_2,
    TEXT_3,
    WARN,
)

LO_COLOR = (120, 150, 200)
NLO_COLOR = GOLD
# One color per generator variant, in the order they are listed.
VARIANT_ORDER = ("lo", "nlo", "nlo_ms", "ms_tuned", "fxfx", "fxfx_tuned")
VARIANT_COLORS = {
    "lo": LO_COLOR,
    "nlo": GOLD,
    "nlo_ms": (120, 220, 170),
    "ms_tuned": (80, 200, 255),
    "fxfx": (230, 130, 200),
    "fxfx_tuned": (255, 110, 110),
}


def variant_color(entry: dict):
    return VARIANT_COLORS.get(entry.get("variant", "lo"), TEXT_2)


def user_results_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "SMLab", "validation")


def _bundled_results() -> str:
    root = getattr(sys, "_MEIPASS", None) if getattr(sys, "frozen", False) else None
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "hep", "validation", "results.json")


def _read_results(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data.get("benchmarks"), dict) else None


def load_results() -> tuple[dict | None, str]:
    """The bundled baseline, overlaid entry by entry with any re-run in the user's folder.

    A re-run may cover only some benchmarks (or be interrupted), so it updates the entries
    it has and leaves the rest of the baseline in place. Each measurement's citation comes
    from the bundled results, whose references are the verified ones.
    """
    user_path, bundled_path = os.path.join(user_results_dir(), "results.json"), _bundled_results()
    bundled, user = _read_results(bundled_path), _read_results(user_path)
    if user is None:
        return bundled, bundled_path if bundled else ""
    if bundled is None:
        return user, user_path
    references = {entry.get("benchmark", key): entry.get("reference")
                  for key, entry in bundled["benchmarks"].items() if entry.get("reference")}
    merged = {**bundled["benchmarks"], **user["benchmarks"]}
    for entry in merged.values():
        reference = references.get(entry.get("benchmark"))
        if reference:
            entry["reference"] = reference
    return {**bundled, "generated": user.get("generated", bundled.get("generated")), "benchmarks": merged}, \
        f"{user_path} + bundled baseline"


def grade_color(chi2: float | None):
    if chi2 is None or not math.isfinite(chi2):
        return TEXT_3
    if chi2 < 2.0:
        return GOOD
    if chi2 < 5.0:
        return WARN
    return BAD


def grouped(results: dict) -> list[tuple[str, list[dict]]]:
    """(benchmark key, generator variants in a fixed order) in the order written."""
    order: list[str] = []
    rows: dict[str, list[dict]] = {}
    for key, value in results.get("benchmarks", {}).items():
        base = value.get("benchmark", key.removesuffix("_nlo"))
        if base not in rows:
            rows[base] = []
            order.append(base)
        rows[base].append(value)
    rank = {name: i for i, name in enumerate(VARIANT_ORDER)}
    return [(base, sorted(rows[base], key=lambda e: rank.get(e.get("variant", "lo"), 99))) for base in order]


class ValidationRun:
    """Re-runs hep/validate.py in WSL in the background."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        from smlab.engine import ENV_PYTHON, WSL_DISTRO, _children_die_with_us, worker_path_in_wsl

        os.makedirs(user_results_dir(), exist_ok=True)
        script = worker_path_in_wsl().replace("worker.py", "validate.py")
        out_dir = user_results_dir()
        drive, rest = os.path.splitdrive(out_dir)
        out_wsl = "/mnt/" + drive[0].lower() + rest.replace("\\", "/") if drive else out_dir
        try:
            _children_die_with_us()
        except Exception:
            pass
        self.lines = []
        self.proc = subprocess.Popen(
            ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", f"exec {ENV_PYTHON} -u {script} --out '{out_wsl}'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        import threading

        def pump():
            for line in self.proc.stdout:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.lines.append(line)

        threading.Thread(target=pump, daemon=True).start()


def draw_validation(app, panel: pygame.Rect) -> None:
    """Contents of the Validation modal. ``app`` supplies the widgets and state."""
    data = app.validation_data
    x = panel.x + 24
    y = panel.y + 84
    if not data:
        app._wrap_block(
            x, y, panel.w - 48,
            "No validation results yet. Run the validation (about 15 minutes on all CPU cores): it generates the "
            "benchmark samples, analyses them with Rivet, and compares with the published measurements.",
            "body", TEXT_2, 20,
        )
        _run_button(app, panel)
        return
    rows = grouped(data)
    list_w = 420
    left = pygame.Rect(x, y, list_w, panel.bottom - y - 70)
    right = pygame.Rect(left.right + 20, y, panel.right - 24 - (left.right + 20), panel.bottom - y - 70)
    app._blit("caption", f"RIVET 4.1 · GENERATED {data.get('generated', '')}", TEXT_3, (x, y - 22))
    row_y = left.y
    for base, entries in rows:
        head = entries[0]
        box = pygame.Rect(left.x, row_y, left.w, 44 + 22 * len(entries))
        selected = app.validation_selected == base
        hover = app._hovered(box)
        pygame.draw.rect(app.screen, ACCENT_SOFT if selected else (SURFACE_3 if hover else SURFACE_2), box, border_radius=10)
        if selected:
            pygame.draw.rect(app.screen, ACCENT, box, 1, border_radius=10)
        app._blit("h2", app.text.fit("h2", head["title"], box.w - 24), TEXT, (box.x + 12, box.y + 8))
        app._blit("caption", app.text.fit("caption", f"{head['tests']} · {', '.join(head['analyses'])}", box.w - 24),
                  TEXT_3, (box.x + 12, box.y + 28))
        ly = box.y + 46
        for entry in entries:
            chi = entry.get("median_chi2_ndf")
            pygame.draw.circle(app.screen, variant_color(entry), (box.x + 17, ly + 8), 4)
            app._blit("small", app.text.fit("small", entry["generator"], 200), TEXT_2, (box.x + 28, ly))
            chi_text = "—" if chi is None else f"χ²/ndf {chi:.1f}"
            shape = entry.get("median_shape_chi2_ndf")
            ratio = entry.get("median_norm_ratio")
            detail = f"shape {shape:.1f} · rate {ratio:.2f}" if shape is not None and ratio is not None else ""
            app._blit("mono_small", chi_text, grade_color(chi), (box.right - 12, ly + 1), "topright")
            if detail:
                app._blit("caption", detail, TEXT_3, (box.right - 110, ly + 3), "topright")
            ly += 22
        app.hot.append((box, "val-select", base))
        row_y = box.bottom + 8
    app._blit("caption", "χ²/ndf: green < 2, amber < 5, red ≥ 5 · rate = MC/data integral", TEXT_3, (left.x, left.bottom + 6))
    selected = next((entries for base, entries in rows if base == app.validation_selected), None)
    if selected is None and rows:
        app.validation_selected = rows[0][0]
        selected = rows[0][1]
    if selected:
        _plot_panel(app, right, selected)
    _run_button(app, panel)


def _run_button(app, panel: pygame.Rect) -> None:
    runner = app.validation_run
    if runner.running:
        label = "Validating… " + (runner.lines[-1][:60] if runner.lines else "")
    else:
        label = "Re-run validation (≈ 15 min, all CPU cores)"
    rect = pygame.Rect(panel.x + 24, panel.bottom - 50, 420, 34)
    app._button(rect, app.text.fit("small", label, 400), "val-run", kind="ghost")
    source = app.validation_path
    if source:
        app._blit("caption", app.text.fit("caption", f"results: {source}", panel.w - 500), TEXT_3,
                  (rect.right + 16, rect.centery), "midleft")


def _plot_panel(app, rect: pygame.Rect, entries: list[dict]) -> None:
    head = entries[0]
    plots = head["plots"]
    if not plots:
        app._wrap_block(rect.x, rect.y, rect.w, "No comparable distributions.", "small", TEXT_2)
        return
    index = app.validation_plot % len(plots)
    plot = plots[index]
    series = []
    for entry in entries:
        match = next((p for p in entry["plots"] if p["path"] == plot["path"]), None)
        if match is not None:
            series.append((entry, match))
    pygame.draw.rect(app.screen, BG, rect, border_radius=12)
    pygame.draw.rect(app.screen, BORDER, rect, 1, border_radius=12)
    title = _latex_lite(plot.get("title") or plot["path"])
    if title.startswith("doi:") and plot.get("ylabel"):
        title = _latex_lite(plot["ylabel"])
    app._blit("h2", app.text.fit("h2", title, rect.w - 190), TEXT, (rect.x + 14, rect.y + 10))
    app._blit("caption", app.text.fit("caption", f"{plot['path']} · {head['reference']}", rect.w - 190), TEXT_3,
              (rect.x + 14, rect.y + 32))
    prev_rect = pygame.Rect(rect.right - 170, rect.y + 10, 36, 30)
    next_rect = pygame.Rect(rect.right - 46, rect.y + 10, 36, 30)
    app._button(prev_rect, "‹", "val-plot", -1, kind="ghost")
    app._button(next_rect, "›", "val-plot", +1, kind="ghost")
    app._blit("small", f"{index + 1} / {len(plots)}", TEXT_2, ((prev_rect.right + next_rect.x) // 2, prev_rect.centery), "center")
    # Legend: data, then each generator with its χ² for this distribution (two per line).
    lx, ly = rect.x + 14, rect.y + 52
    items = [("Data", TEXT, None, False)] + [
        (entry["generator"], variant_color(entry), match["chi2_ndf"], mc_limited(match)) for entry, match in series]
    limited = any(flag for *_rest, flag in items)
    for label, color, chi, flag in items:
        text = label if chi is None else f"{label}  {chi:.1f}{'*' if flag else ''}"
        width = app.text.size("caption", text)[0] + 30
        if lx + width > rect.right - 10:
            lx, ly = rect.x + 14, ly + 16
        pygame.draw.line(app.screen, color, (lx, ly + 7), (lx + 14, ly + 7), 3)
        app._blit("caption", text, TEXT_2, (lx + 18, ly))
        lx += width
    if limited:
        ly += 16
        app._blit("caption", "* χ² limited by MC statistics: the simulation's error exceeds the data's", WARN,
                  (rect.x + 14, ly))
    top = ly + 22
    main = pygame.Rect(rect.x + 70, top + 8, rect.w - 90, int((rect.bottom - top - 60) * 0.7))
    ratio = pygame.Rect(main.x, main.bottom + 12, main.w, rect.bottom - 34 - (main.bottom + 12))
    _draw_distribution(app, main, ratio, plot, [(match["mc"], match["mc_err"], variant_color(entry)) for entry, match in series])
    xlabel = plot.get("xlabel", "")
    if xlabel:
        app._blit("caption", app.text.fit("caption", _latex_lite(xlabel), rect.w - 40), TEXT_3, (main.centerx, rect.bottom - 16), "center")


def mc_limited(plot: dict) -> bool:
    """True when the χ² mostly measures the simulation's own statistical noise: a bin with data
    but no simulated events, or an MC error larger than the data error in most bins."""
    rows = [(m, me, de) for d, m, me, de in zip(plot["data"], plot["mc"], plot["mc_err"], plot["data_err"]) if d]
    if any(m == 0.0 and me == 0.0 for m, me, _de in rows):
        return True
    ratios = sorted(me / de for _m, me, de in rows if de > 0.0)
    return bool(ratios) and ratios[len(ratios) // 2] > 1.0


def _latex_lite(text: str) -> str:
    for old, new in (("$", ""), ("\\mathrm", ""), ("{", ""), ("}", ""), ("\\eta", "η"), ("\\phi", "φ"),
                     ("\\text", ""), ("_T", "T"), ("^\\ell", "ℓ"), ("\\ell", "ℓ"), ("\\", "")):
        text = text.replace(old, new)
    return text


def _draw_distribution(app, main: pygame.Rect, ratio: pygame.Rect, plot: dict, series: list) -> None:
    edges = plot["edges"]
    data, derr = plot["data"], plot["data_err"]
    values = [v for v in data if v > 0] + [v for mc, _e, _c in series for v in mc if v > 0]
    positive = values and min(values) > 0
    lo = min(values) if values else 0.0
    hi = max(values) if values else 1.0
    log_y = positive and hi / max(lo, 1e-300) > 200.0
    span_x = edges[-1] - edges[0]
    log_x = edges[0] > 0 and edges[-1] / edges[0] > 50.0

    def fx(value: float) -> float:
        if log_x:
            return main.x + (math.log(value) - math.log(edges[0])) / (math.log(edges[-1]) - math.log(edges[0])) * main.w
        return main.x + (value - edges[0]) / (span_x or 1.0) * main.w

    if log_y:
        ymin, ymax = math.log10(lo) - 0.2, math.log10(hi) + 0.3

        def fy(value: float) -> float:
            v = math.log10(max(value, 10 ** ymin))
            return main.bottom - (v - ymin) / (ymax - ymin) * main.h
    else:
        ymax = hi * 1.15 if hi > 0 else 1.0
        ymin = min(0.0, min(data + [v for mc, _e, _c in series for v in mc]))

        def fy(value: float) -> float:
            return main.bottom - (value - ymin) / ((ymax - ymin) or 1.0) * main.h

    pygame.draw.rect(app.screen, (14, 18, 29), main)
    pygame.draw.rect(app.screen, BORDER, main, 1)
    previous = app.screen.get_clip()
    app.screen.set_clip(main.inflate(2, 2))
    for mc, _err, color in series:
        for i in range(len(edges) - 1):
            x0, x1 = fx(edges[i]), fx(edges[i + 1])
            y = fy(mc[i])
            pygame.draw.line(app.screen, color, (x0, y), (x1, y), 2)
            if i + 1 < len(mc):
                pygame.draw.line(app.screen, color, (x1, y), (x1, fy(mc[i + 1])), 1)
    for i in range(len(edges) - 1):
        cx = (fx(edges[i]) + fx(edges[i + 1])) / 2
        top, bottom = fy(data[i] + derr[i]), fy(max(data[i] - derr[i], 1e-300 if log_y else data[i] - derr[i]))
        pygame.draw.line(app.screen, TEXT, (cx, top), (cx, bottom), 1)
        pygame.draw.circle(app.screen, TEXT, (int(cx), int(fy(data[i]))), 3)
    app.screen.set_clip(previous)
    top_label = f"{hi:.3g}" if not log_y else f"1e{ymax:.0f}"
    app._blit("mono_small", top_label, TEXT_3, (main.x - 6, main.y), "topright")
    app._blit("mono_small", "log y" if log_y else "", TEXT_3, (main.x - 6, main.bottom - 14), "topright")
    # Ratio MC / data with the data's relative uncertainty band.
    pygame.draw.rect(app.screen, (14, 18, 29), ratio)
    pygame.draw.rect(app.screen, BORDER, ratio, 1)
    r_lo, r_hi = 0.5, 1.5

    def fr(value: float) -> float:
        value = min(max(value, r_lo), r_hi)
        return ratio.bottom - (value - r_lo) / (r_hi - r_lo) * ratio.h

    for i in range(len(edges) - 1):
        if data[i] <= 0:
            continue
        band = derr[i] / data[i]
        x0, x1 = fx(edges[i]), fx(edges[i + 1])
        pygame.draw.rect(app.screen, (40, 46, 64), pygame.Rect(x0, fr(1 + band), max(1, x1 - x0), max(1, fr(1 - band) - fr(1 + band))))
    pygame.draw.line(app.screen, TEXT_3, (ratio.x, fr(1.0)), (ratio.right, fr(1.0)), 1)
    for mc, _err, color in series:
        for i in range(len(edges) - 1):
            if data[i] <= 0:
                continue
            x0, x1 = fx(edges[i]), fx(edges[i + 1])
            pygame.draw.line(app.screen, color, (x0, fr(mc[i] / data[i])), (x1, fr(mc[i] / data[i])), 2)
    app._blit("mono_small", "MC/data", TEXT_3, (ratio.x - 6, ratio.y), "topright")
    app._blit("mono_small", "1.5", TEXT_3, (ratio.x - 6, ratio.y + 12), "topright")
    app._blit("mono_small", "0.5", TEXT_3, (ratio.x - 6, ratio.bottom - 14), "topright")
    app._blit("mono_small", f"{edges[0]:g}", TEXT_3, (main.x, ratio.bottom + 3))
    app._blit("mono_small", f"{edges[-1]:g}", TEXT_3, (main.right, ratio.bottom + 3), "topright")
