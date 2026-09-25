"""Colors, type, and drawing primitives shared by every panel."""

from __future__ import annotations

import os

import pygame

# Surfaces, darkest first.
BG = (10, 13, 21)
SURFACE = (16, 20, 31)
SURFACE_2 = (22, 27, 41)
SURFACE_3 = (31, 37, 55)
BORDER = (38, 45, 66)
BORDER_STRONG = (60, 70, 98)

# Text.
TEXT = (234, 238, 246)
TEXT_2 = (168, 177, 198)
TEXT_3 = (114, 123, 148)

# One accent, plus status colors.
ACCENT = (92, 168, 255)
ACCENT_HOVER = (122, 186, 255)
ACCENT_SOFT = (30, 48, 80)
ACCENT_INK = (8, 14, 26)
GOOD = (70, 210, 150)
WARN = (255, 188, 88)
BAD = (255, 108, 124)
WHITE = (246, 248, 252)

# Names used by the detector, 3D view, and report sheet.
BG_RAISED = SURFACE_2
PANEL = SURFACE
PANEL_EDGE = BORDER
HAIRLINE = BORDER
GOLD = (236, 196, 110)  # the solenoid coil and the collision point
GOLD_DIM = (126, 104, 62)
MUTED = TEXT_2
DIM = TEXT_3
CYAN = ACCENT

_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


def _font(files: tuple[str, ...], fallback: str, size: int, bold: bool = False) -> pygame.font.Font:
    for name in files:
        path = os.path.join(_FONT_DIR, name)
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    path = pygame.font.match_font(fallback, bold=bold)
    if path:
        return pygame.font.Font(path, size)
    return pygame.font.Font(None, size + 4)


def load_fonts() -> dict[str, pygame.font.Font]:
    regular = ("segoeui.ttf",)
    semibold = ("seguisb.ttf", "segoeuib.ttf")
    mono = ("consola.ttf",)
    mono_bold = ("consolab.ttf", "consola.ttf")
    return {
        "brand": _font(semibold, "segoeui", 20, True),
        "h1": _font(semibold, "segoeui", 22, True),
        "h2": _font(semibold, "segoeui", 15, True),
        "body": _font(regular, "segoeui", 14),
        "small": _font(regular, "segoeui", 13),
        "caption": _font(semibold, "segoeui", 11, True),
        "big": _font(mono_bold, "consolas", 26, True),
        # Keys used by the detector, 3D view, and report sheet.
        "title": _font(semibold, "segoeui", 22, True),
        "subtitle": _font(regular, "segoeui", 13),
        "section": _font(semibold, "segoeui", 11, True),
        "ui": _font(regular, "segoeui", 14),
        "ui_small": _font(regular, "segoeui", 13),
        "mono": _font(mono, "consolas", 14),
        "mono_small": _font(mono, "consolas", 12),
        "label": _font(semibold, "segoeui", 13, True),
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


class TextCache:
    """Rendered strings keyed by font, text, and color. Cleared when it grows large."""

    def __init__(self, fonts: dict[str, pygame.font.Font]):
        self.fonts = fonts
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(self, key: str, text: str, color) -> pygame.Surface:
        entry = (key, text, tuple(color))
        image = self._cache.get(entry)
        if image is None:
            if len(self._cache) > 3000:
                self._cache.clear()
            image = self.fonts[key].render(text, True, color)
            self._cache[entry] = image
        return image

    def size(self, key: str, text: str) -> tuple[int, int]:
        return self.fonts[key].size(text)

    def wrap(self, key: str, text: str, width: int) -> list[str]:
        font = self.fonts[key]
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
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

    def fit(self, key: str, text: str, width: int) -> str:
        """Shorten ``text`` with an ellipsis until it fits ``width`` pixels."""
        font = self.fonts[key]
        if font.size(text)[0] <= width:
            return text
        shown = text
        while len(shown) > 1 and font.size(shown + "…")[0] > width:
            shown = shown[:-1]
        return shown.rstrip() + "…"


def fill_round(surf: pygame.Surface, rect: pygame.Rect, color, radius: int = 10) -> None:
    """Rounded rectangle; ``color`` may carry an alpha channel."""
    if len(color) == 4 and color[3] < 255:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, color, layer.get_rect(), border_radius=radius)
        surf.blit(layer, rect.topleft)
    else:
        pygame.draw.rect(surf, color[:3], rect, border_radius=radius)


def card(surf: pygame.Surface, rect: pygame.Rect, *, fill=SURFACE, border=BORDER, radius: int = 14) -> None:
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, 1, border_radius=radius)


def shadow(surf: pygame.Surface, rect: pygame.Rect, radius: int = 16, spread: int = 14, alpha: int = 120) -> None:
    layer = pygame.Surface((rect.w + spread * 2, rect.h + spread * 2), pygame.SRCALPHA)
    for step in range(spread, 0, -2):
        a = int(alpha * (1.0 - step / spread) ** 2 * 0.5)
        pygame.draw.rect(
            layer,
            (0, 0, 0, a),
            pygame.Rect(spread - step, spread - step + 4, rect.w + step * 2, rect.h + step * 2),
            border_radius=radius + step,
        )
    surf.blit(layer, (rect.x - spread, rect.y - spread))
