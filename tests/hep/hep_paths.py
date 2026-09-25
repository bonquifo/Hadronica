"""Paths shared by the WSL-side tests."""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NLO_DIR = os.path.expanduser("~/smlab-cache/nlo")
MG5_ENV = os.path.expanduser("~/micromamba/envs/smlab-mg5")


def sample_path(name: str) -> str:
    """events.lhe of a generated sample, or skip the test if it was not generated."""
    path = os.path.join(NLO_DIR, name, "events.lhe")
    if not os.path.exists(path):
        pytest.skip(f"sample {name} not generated (hep/mg5/produce_all.sh)")
    return path
