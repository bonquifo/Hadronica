"""Generate Hadronica's icons: hadronica/assets/icon.ico (the exe), window_icon.png (the window), icon.png (512 px).

    python tools/make_icon.py

The motif is the end-on event display the application draws: a collision vertex
at the centre of a detector, charged tracks curving out of it in the solenoid
field (circles through the vertex, ending on the calorimeter), in the
application's own theme colours. Each size is drawn at four times its
resolution and downsampled. Sizes up to 32 px use a simplified drawing (three
bold tracks, no glow, a larger vertex) so they stay legible in the title bar
and taskbar. Also writes tools/icon_preview.png, a contact sheet for review.
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hadronica", "assets")

# hadronica/theme.py
BG = (10, 13, 21)
RIM = (30, 38, 58)
ACCENT = (92, 168, 255)
ACCENT_SOFT = (34, 54, 90)
GOOD = (70, 210, 150)
BAD = (255, 108, 124)
GOLD = (236, 196, 110)
GOLD_DIM = (140, 116, 70)
WHITE = (255, 255, 255)

# Tracks: (angle where the track reaches the calorimeter, in degrees counter-clockwise from +x;
# signed radius of curvature in barrel radii, positive bending counter-clockwise; colour).
# Exit points are spaced evenly around the ring; low-momentum tracks curl more.
TRACKS = (
    (25.0, 2.60, ACCENT),
    (95.0, 0.95, ACCENT),
    (165.0, -0.80, GOOD),
    (240.0, 1.40, GOLD),
    (315.0, -0.70, BAD),
)
SMALL_TRACKS = (
    (90.0, 0.85, ACCENT),
    (210.0, -0.85, GOOD),
    (330.0, 0.85, BAD),
)


def launch_angle(exit_deg: float, curvature: float) -> float:
    """Initial direction of a track through the vertex that meets the barrel at ``exit_deg``.

    A circle of radius R through the vertex crosses the barrel (radius r) a chord of length r
    away, turned by asin(r / 2R) from its initial direction, in the sense it bends.
    """
    turn = math.degrees(math.asin(min(1.0, 1.0 / (2.0 * abs(curvature)))))
    return exit_deg - math.copysign(turn, curvature)


def track(center, barrel, direction_deg, curvature, step):
    """Points along a circular track from the vertex until it reaches the barrel radius."""
    cx, cy = center
    phi = math.radians(direction_deg)
    ux, uy = math.cos(phi), -math.sin(phi)  # screen y points down
    radius = abs(curvature) * barrel
    bend = 1.0 if curvature > 0 else -1.0  # +1: counter-clockwise on screen
    # The circle's centre lies perpendicular to the initial direction, on the side it bends toward.
    nx, ny = uy * bend, -ux * bend
    ox, oy = cx + nx * radius, cy + ny * radius
    start = math.atan2(cy - oy, cx - ox)
    sense = 1.0 if (-math.sin(start) * ux + math.cos(start) * uy) > 0 else -1.0
    points, arc = [], 0.0
    while arc < math.pi * radius:  # at most half a turn: the track then leaves or curls back
        angle = start + sense * arc / radius
        x, y = ox + radius * math.cos(angle), oy + radius * math.sin(angle)
        if math.hypot(x - cx, y - cy) >= barrel:
            break
        points.append((x, y))
        arc += step
    return points


def stroke(draw: ImageDraw.ImageDraw, points, width: float, colour) -> None:
    """A round brush stamped densely along the path: smooth, even, with round caps."""
    r = width / 2
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)


def draw_icon(size: int, small: bool | None = None) -> Image.Image:
    n = size * 4
    small = size <= 32 if small is None else small
    c = (n / 2, n / 2)
    image = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)

    corner = n * 0.22
    d.rounded_rectangle((0, 0, n - 1, n - 1), radius=corner, fill=BG + (255,))
    if not small:
        inset = n * 0.012
        d.rounded_rectangle((inset, inset, n - 1 - inset, n - 1 - inset), radius=corner - inset,
                            outline=RIM + (255,), width=max(1, round(n * 0.014)))

    barrel = n * (0.37 if small else 0.355)
    ring = n * (0.06 if small else 0.028)
    d.ellipse((c[0] - barrel, c[1] - barrel, c[0] + barrel, c[1] + barrel), outline=GOLD_DIM + (255,),
              width=max(2, round(ring)))
    if not small:
        for frac in (0.42, 0.68):
            r = barrel * frac
            d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), outline=ACCENT_SOFT + (255,),
                      width=max(1, round(n * 0.010)))

    width = n * (0.085 if small else 0.036)
    inner = barrel - ring / 2  # tracks end on the inside of the calorimeter ring
    paths = [(track(c, inner, launch_angle(a, k), k, width / 4), colour)
             for a, k, colour in (SMALL_TRACKS if small else TRACKS)]

    if not small:
        glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        for points, colour in paths:
            stroke(g, points, width * 2.6, colour + (110,))
        image = Image.alpha_composite(image, glow.filter(ImageFilter.GaussianBlur(n * 0.018)))
        d = ImageDraw.Draw(image)
    for points, colour in paths:
        stroke(d, points, width, colour + (255,))
        if not small and points:
            x, y = points[-1]
            r = width * 0.95
            d.ellipse((x - r, y - r, x + r, y + r), fill=colour + (255,))

    if not small:
        halo = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        hr = n * 0.10
        ImageDraw.Draw(halo).ellipse((c[0] - hr, c[1] - hr, c[0] + hr, c[1] + hr), fill=GOLD + (200,))
        image = Image.alpha_composite(image, halo.filter(ImageFilter.GaussianBlur(n * 0.03)))
        d = ImageDraw.Draw(image)
    core = n * (0.095 if small else 0.052)
    d.ellipse((c[0] - core, c[1] - core, c[0] + core, c[1] + core), fill=WHITE + (255,))

    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=corner, fill=255)
    image.putalpha(ImageChops.darker(image.getchannel("A"), mask))
    return image.resize((size, size), Image.LANCZOS)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sizes = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
    frames = {size: draw_icon(size) for size in sizes}
    frames[256].save(os.path.join(OUT, "icon.ico"), format="ICO", sizes=[(s, s) for s in sizes],
                     append_images=[frames[s] for s in sizes if s != 256])
    draw_icon(512).save(os.path.join(OUT, "icon.png"))
    # The window icon: one image that Windows scales to the title bar (16-24 px) and taskbar
    # (32-48 px), so it uses the simplified small-size drawing.
    draw_icon(64, small=True).save(os.path.join(OUT, "window_icon.png"))
    # Contact sheet: every size on light and dark backgrounds, plus 16/32 px enlarged 4×.
    width = sum(sizes) + 14 * (len(sizes) + 1)
    sheet = Image.new("RGBA", (width + 300, 2 * 270 + 10), (0, 0, 0, 255))
    for row, background in enumerate(((236, 236, 236, 255), (32, 32, 36, 255))):
        top = row * 275
        ImageDraw.Draw(sheet).rectangle((0, top, width + 300, top + 265), fill=background)
        x = 14
        for size in sizes:
            sheet.alpha_composite(frames[size], (x, top + 8 + (256 - size) // 2))
            x += size + 14
        for i, size in enumerate((16, 32)):
            big = frames[size].resize((size * 4, size * 4), Image.NEAREST)
            sheet.alpha_composite(big, (x + 10 + i * 80, top + 60))
    sheet.save(os.path.join(ROOT, "tools", "icon_preview.png"))
    print("wrote icon.ico, icon.png and window_icon.png in", OUT)


if __name__ == "__main__":
    main()
