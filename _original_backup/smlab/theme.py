"""Colors and type used by the event display."""

from __future__ import annotations

import pygame

BG = (7, 10, 18)
BG_RAISED = (12, 18, 32)
PANEL = (10, 16, 30)
PANEL_EDGE = (36, 64, 102)
HAIRLINE = (28, 48, 78)
GOLD = (228, 195, 122)
GOLD_DIM = (120, 98, 58)
TEXT = (230, 237, 246)
MUTED = (142, 160, 184)
DIM = (88, 104, 128)
CYAN = (122, 215, 255)
GOOD = (61, 220, 151)
WARN = (255, 186, 92)
BAD = (255, 107, 122)
WHITE = (246, 248, 252)


def load_fonts() -> dict[str, pygame.font.Font]:
    def face(name: str, size: int, bold: bool = False) -> pygame.font.Font:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
        return pygame.font.SysFont(name, size, bold=bold)

    return {
        "title": face("segoeui", 28, bold=True),
        "subtitle": face("segoeui", 14),
        "section": face("segoeui", 12, bold=True),
        "ui": face("segoeui", 15),
        "ui_small": face("segoeui", 13),
        "mono": face("consolas", 14),
        "mono_small": face("consolas", 12),
        "label": face("segoeui", 13, bold=True),
    }


def particle_color(pdg: int) -> tuple[int, int, int]:
    return {
        11: (88, 214, 255),
        12: (168, 180, 198),
        13: (132, 164, 255),
        14: (168, 180, 198),
        15: (196, 146, 255),
        16: (168, 180, 198),
        1: (72, 214, 146),
        2: (255, 106, 128),
        3: (255, 150, 196),
        4: (48, 214, 214),
        5: (176, 146, 255),
        6: (255, 128, 168),
        21: (255, 186, 72),
        22: (255, 228, 120),
        23: (255, 164, 78),
        24: (72, 230, 176),
        25: (255, 244, 220),
        111: (255, 214, 140),
        211: (255, 176, 96),
    }.get(abs(pdg), (210, 220, 235))


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
