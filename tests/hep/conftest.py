"""WSL-side tests: real PYTHIA 8, pyhepmc, and the MadGraph5_aMC@NLO samples in ~/smlab-cache."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "hep"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
