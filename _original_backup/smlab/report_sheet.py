"""The collision report: what was produced, and energy across the slowed picture."""

from __future__ import annotations

import pygame

from smlab.report import EnergyTimeline, english_name, format_duration, format_energy, outcome_line, report_cards
from smlab.theme import DIM, GOLD, GOLD_DIM, HAIRLINE, MUTED, PANEL, TEXT, particle_color


def draw_report_sheet(surf: pygame.Surface, fonts: dict, event, timeline: EnergyTimeline) -> pygame.Rect:
    """Draw the report and return the close-button rectangle."""
    shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    shade.fill((4, 6, 12, 210))
    surf.blit(shade, (0, 0))
    panel = pygame.Rect(0, 0, min(1180, surf.get_width() - 48), min(760, surf.get_height() - 36))
    panel.center = surf.get_rect().center
    pygame.draw.rect(surf, (8, 12, 22), panel, border_radius=16)
    pygame.draw.rect(surf, GOLD, panel, 1, border_radius=16)

    title = fonts["title"].render("Collision report", True, TEXT)
    surf.blit(title, (panel.x + 28, panel.y + 16))
    close = pygame.Rect(panel.right - 108, panel.y + 18, 80, 30)
    pygame.draw.rect(surf, (18, 28, 46), close, border_radius=8)
    pygame.draw.rect(surf, GOLD, close, 1, border_radius=8)
    label = fonts["ui"].render("Close", True, TEXT)
    surf.blit(label, label.get_rect(center=close.center))

    headline = _headline(event)
    surf.blit(fonts["ui"].render(headline, True, GOLD), (panel.x + 28, panel.y + 52))
    formed = format_duration(timeline.formed_s)
    sub = (
        f"Hard collision {format_energy(timeline.hard_gev)}"
        f"    picture {timeline.end_s:.1f} s, slowed"
        f"    collision itself {formed}"
    )
    surf.blit(fonts["ui_small"].render(sub, True, MUTED), (panel.x + 28, panel.y + 76))

    body = pygame.Rect(panel.x + 20, panel.y + 108, panel.w - 40, panel.h - 148)
    left = pygame.Rect(body.x, body.y, int(body.w * 0.40), body.h)
    right = pygame.Rect(left.right + 12, body.y, body.w - left.w - 12, body.h)
    _produced(surf, fonts, left, event, timeline)
    _graphs(surf, fonts, right, timeline)

    note = fonts["mono_small"].render(
        "Energies stay constant while particles travel. They change when the particles meet. The picture is slowed.",
        True,
        DIM,
    )
    surf.blit(note, (panel.x + 28, panel.bottom - 28))
    return close


def _headline(event) -> str:
    cards = dict(report_cards(event))
    produced = cards.get("Produced", ("products",))[0]
    who = cards.get("Produced", ("", "the two particles"))[1]
    return f"{produced}  ·  {who}"


def _produced(surf, fonts, rect, event, timeline: EnergyTimeline) -> None:
    surf.blit(fonts["section"].render("WHAT WAS PRODUCED", True, GOLD), (rect.x, rect.y))
    y = rect.y + 24
    rows = [particle for particle in event.particles if particle.status != "beam"]
    for particle in rows[:8]:
        name = "radiated photon" if particle.status == "isr" else _name(particle)
        energy = format_energy(particle.p4.e)
        fate = outcome_line(particle)
        color = particle_color(particle.pdg)
        pygame.draw.circle(surf, color, (rect.x + 6, y + 7), 4)
        surf.blit(fonts["ui_small"].render(name, True, TEXT), (rect.x + 16, y))
        detail = fonts["mono_small"].render(f"{energy}   {fate}", True, MUTED)
        surf.blit(detail, (rect.x + 16, y + 16))
        y += 38
        if y > rect.bottom - 90:
            break
    if len(rows) > 8:
        surf.blit(fonts["ui_small"].render("Further products are in More → Every product.", True, DIM), (rect.x, y))
        y += 22
    surf.blit(fonts["section"].render("RADIATION", True, GOLD), (rect.x, y + 6))
    y += 28
    for line in _wrap(fonts["ui_small"], timeline.radiation, rect.w - 8):
        surf.blit(fonts["ui_small"].render(line, True, TEXT), (rect.x, y))
        y += 16


def _name(particle) -> str:
    return english_name(particle.pdg)


def _graphs(surf, fonts, rect, timeline: EnergyTimeline) -> None:
    gap = 8
    height = (rect.h - gap * 2) // 3
    boxes = [
        pygame.Rect(rect.x, rect.y + index * (height + gap), rect.w, height) for index in range(3)
    ]
    _carrier_graph(surf, fonts, boxes[0], timeline)
    _total_graph(surf, fonts, boxes[1], timeline)
    _split_graph(surf, fonts, boxes[2], timeline)


def _carrier_graph(surf, fonts, rect, timeline: EnergyTimeline) -> None:
    peak = max(timeline.total, sum(item.energy for item in timeline.after), 1e-6) * 1.05
    _frame(surf, fonts, rect, "Energy of each particle", "stacked, from the start of the picture to the end")
    plot = _plot_rect(rect)
    _stack(surf, plot, 0.0, timeline.meet_s, timeline.before, peak, timeline)
    _stack(surf, plot, timeline.meet_s, timeline.end_s, timeline.after, peak, timeline)
    _meet(surf, plot, timeline)
    _legend(surf, fonts, plot, timeline)


def _total_graph(surf, fonts, rect, timeline: EnergyTimeline) -> None:
    after = sum(item.energy for item in timeline.after)
    peak = max(timeline.total, after, 1e-6) * 1.15
    _frame(surf, fonts, rect, "Total energy", "the same before and after they meet")
    plot = _plot_rect(rect)
    _meet(surf, plot, timeline)
    _step(surf, plot, 0.0, timeline.end_s, timeline.total, peak, timeline, (122, 215, 255))
    label = fonts["mono_small"].render(format_energy(timeline.total), True, (122, 215, 255))
    surf.blit(label, (plot.x + 8, plot.y + 6))
    surf.blit(fonts["mono_small"].render("start", True, DIM), (plot.x, plot.bottom + 2))
    meet = fonts["mono_small"].render("meet", True, GOLD_DIM)
    meet_x = plot.x + timeline.meet_s / timeline.end_s * plot.w
    surf.blit(meet, (meet_x - meet.get_width() / 2, plot.bottom + 2))
    end = fonts["mono_small"].render("end", True, DIM)
    surf.blit(end, (plot.right - end.get_width(), plot.bottom + 2))


def _split_graph(surf, fonts, rect, timeline: EnergyTimeline) -> None:
    rest_before = sum(item.rest for item in timeline.before)
    kin_before = sum(item.kinetic for item in timeline.before)
    rest_after = sum(item.rest for item in timeline.after)
    kin_after = sum(item.kinetic for item in timeline.after)
    peak = max(rest_before, kin_before, rest_after, kin_after, 1e-6) * 1.1
    _frame(surf, fonts, rect, "Rest energy and kinetic energy", "rest is mc²; kinetic is what is left")
    plot = _plot_rect(rect)
    _meet(surf, plot, timeline)
    _step(surf, plot, 0.0, timeline.meet_s, kin_before, peak, timeline, GOLD)
    _step(surf, plot, timeline.meet_s, timeline.end_s, kin_after, peak, timeline, GOLD)
    _step(surf, plot, 0.0, timeline.meet_s, rest_before, peak, timeline, (168, 180, 198))
    _step(surf, plot, timeline.meet_s, timeline.end_s, rest_after, peak, timeline, (168, 180, 198))
    surf.blit(fonts["mono_small"].render("kinetic", True, GOLD), (plot.right - 120, plot.y + 4))
    surf.blit(fonts["mono_small"].render("rest", True, (168, 180, 198)), (plot.right - 52, plot.y + 4))
    if max(rest_before, rest_after) < 0.02 * max(kin_before, kin_after, 1e-9):
        note = fonts["mono_small"].render("rest energy is below this scale", True, DIM)
        surf.blit(note, (plot.x + 8, plot.y + 4))


def _frame(surf, fonts, rect, title, subtitle) -> None:
    pygame.draw.rect(surf, PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, HAIRLINE, rect, 1, border_radius=10)
    surf.blit(fonts["ui_small"].render(title, True, TEXT), (rect.x + 12, rect.y + 6))
    surf.blit(fonts["mono_small"].render(subtitle, True, DIM), (rect.x + 12, rect.y + 22))


def _plot_rect(rect: pygame.Rect) -> pygame.Rect:
    return pygame.Rect(rect.x + 12, rect.y + 42, rect.w - 24, rect.h - 62)


def _meet(surf, plot, timeline: EnergyTimeline) -> None:
    pygame.draw.line(surf, (40, 56, 78), (plot.x, plot.bottom), (plot.right, plot.bottom), 1)
    if timeline.end_s <= 0:
        return
    x = plot.x + timeline.meet_s / timeline.end_s * plot.w
    pygame.draw.line(surf, GOLD_DIM, (x, plot.y), (x, plot.bottom), 1)


def _stack(surf, plot, t0, t1, carriers, peak, timeline) -> None:
    if timeline.end_s <= 0 or peak <= 0 or not carriers:
        return
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    floor = 0.0
    x0 = plot.x + t0 / timeline.end_s * plot.w
    x1 = plot.x + t1 / timeline.end_s * plot.w
    for item in carriers:
        y0 = plot.bottom - floor / peak * (plot.h - 6)
        y1 = plot.bottom - (floor + item.energy) / peak * (plot.h - 6)
        color = (*particle_color(item.pdg), 170)
        pygame.draw.polygon(layer, color, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        floor += item.energy
    surf.blit(layer, (0, 0))


def _step(surf, plot, t0, t1, energy, peak, timeline, color) -> None:
    if timeline.end_s <= 0 or peak <= 0:
        return
    x0 = plot.x + t0 / timeline.end_s * plot.w
    x1 = plot.x + t1 / timeline.end_s * plot.w
    y = plot.bottom - (energy / peak) * (plot.h - 6)
    pygame.draw.line(surf, color, (x0, y), (x1, y), 2)


def _legend(surf, fonts, plot, timeline: EnergyTimeline) -> None:
    x = plot.x
    y = plot.bottom + 2
    seen = []
    for item in timeline.before + timeline.after:
        if item.name in seen:
            continue
        seen.append(item.name)
        if len(seen) > 4:
            break
        pygame.draw.line(surf, particle_color(item.pdg), (x, y + 6), (x + 12, y + 6), 2)
        text = fonts["mono_small"].render(item.name, True, MUTED)
        surf.blit(text, (x + 16, y))
        x += text.get_width() + 28


def _wrap(font, text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if font.size(trial)[0] <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
