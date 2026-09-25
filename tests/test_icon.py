"""The application icon: the exe's multi-size .ico and the window icon."""

from __future__ import annotations

import os

import pygame
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "smlab", "assets")
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def test_exe_icon_has_every_size_windows_asks_for():
    with Image.open(os.path.join(ASSETS, "icon.ico")) as ico:
        assert set(ico.info["sizes"]) == {(s, s) for s in SIZES}
        for size in SIZES:
            ico.size = (size, size)
            frame = ico.convert("RGBA")
            alpha = frame.getchannel("A")
            # A rounded tile: opaque centre, transparent corners.
            assert alpha.getpixel((size // 2, size // 2)) == 255
            assert alpha.getpixel((0, 0)) < 64
            # The bright vertex sits at the centre.
            assert min(frame.getpixel((size // 2, size // 2))[:3]) > 200


def test_window_uses_the_generated_icon():
    from smlab.app import LabApp

    application = LabApp(size=(1480, 900), headless=True, seed=1)
    try:
        icon = application._icon()
        assert icon.get_size() == (64, 64)
        assert min(icon.get_at((32, 32))[:3]) > 220  # the white vertex
        assert icon.get_at((0, 0)).a < 64  # rounded, transparent corner
        # Not the drawn fallback: the generated icon has the coloured tracks (theme blue, green, red).
        pixels = [icon.get_at((x, y))[:3] for x in range(64) for y in range(64)]
        for colour in ((92, 168, 255), (70, 210, 150), (255, 108, 124)):
            assert any(sum(abs(a - b) for a, b in zip(p, colour)) < 60 for p in pixels), colour
    finally:
        pygame.quit()


def test_build_embeds_and_bundles_the_icon():
    with open(os.path.join(ROOT, "build_exe.ps1"), encoding="utf-8") as handle:
        script = handle.read()
    assert '"--icon", "smlab\\assets\\icon.ico"' in script
    assert "smlab\\assets\\window_icon.png;smlab\\assets" in script


@pytest.mark.parametrize("name", ["icon.ico", "icon.png", "window_icon.png"])
def test_assets_exist(name):
    assert os.path.getsize(os.path.join(ASSETS, name)) > 1000
