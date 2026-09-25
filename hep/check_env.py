"""Print the versions of the engines SMLab uses, and where Delphes keeps its cards."""

import glob
import os
import shutil
import sys

import pythia8

# The conda activation script normally sets PYTHIA8DATA; the worker runs without activation.
XMLDIR = os.environ.get("PYTHIA8DATA") or os.path.join(sys.prefix, "share", "Pythia8", "xmldoc")

probe = pythia8.Pythia(XMLDIR, False)
print("python", sys.version.split()[0])
print("pythia", f"{probe.settings.parm('Pythia:versionNumber'):.3f}")
print("pythia info attribute:", type(getattr(probe, "info", None)).__name__, "infoPython:", hasattr(probe, "infoPython"))
try:
    import lhapdf

    print("lhapdf", lhapdf.version(), "sets:", ", ".join(lhapdf.availablePDFSets()[:5]))
except Exception as exc:  # optional
    print("lhapdf unavailable:", exc)
try:
    import fastjet

    print("fastjet", getattr(fastjet, "__version__", "ok"))
except Exception as exc:
    print("fastjet unavailable:", exc)
print("DelphesHepMC3:", shutil.which("DelphesHepMC3", path=os.path.join(sys.prefix, "bin")) or "not found")
prefix = sys.prefix
cards = glob.glob(os.path.join(prefix, "**", "delphes_card_CMS.tcl"), recursive=True)
print("CMS card:", cards[0] if cards else "not found")
