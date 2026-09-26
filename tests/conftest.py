import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# tests/hep runs inside WSL with the hadronica-hep environment (PYTHIA, pyhepmc, Delphes);
# on Windows, tests/test_wsl_suite.py runs it there instead of collecting it here.
try:
    import pythia8  # noqa: F401
except ImportError:
    collect_ignore = ["hep"]
