"""Check Delphes muon reconstruction against the truth record for pp → Z → ℓℓ.

For each event, count truth muons inside the CMS muon acceptance (|η| < 2.4,
pT > 10 GeV) and compare with the muons Delphes reconstructs. A correct
HepMC conversion gives a high, card-level efficiency (≈ 95 % per muon).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import simulate  # noqa: E402
from worker import Worker  # noqa: E402

worker = Worker()
worker.init({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13600.0, "seed": 11})
events = worker.generate(200)["events"]
recos, label = simulate(events, "pp", 13600.0)
in_acc = found = 0
out_acc_events = 0
for raw, reco in zip(events, recos):
    truth = []
    for row in raw["particles"]:
        if row[1] > 0 and abs(row[0]) == 13:
            px, py, pz = row[6], row[7], row[8]
            pt = math.hypot(px, py)
            eta = math.asinh(pz / pt) if pt > 0 else 99.0
            truth.append((pt, eta))
    accepted = [t for t in truth if t[0] > 10.0 and abs(t[1]) < 2.4]
    in_acc += len(accepted)
    reco_mu = [m for m in reco.get("muons", []) if m[0] > 10.0]
    found += min(len(reco_mu), len(accepted))
    if len(truth) >= 1 and not accepted:
        out_acc_events += 1
print(label)
print(f"truth muons in acceptance: {in_acc}, matched reconstructed: {found}, efficiency {found / max(in_acc, 1):.3f}")
print(f"events whose muons all fall outside |eta|<2.4 or pT>10: {out_acc_events} / {len(events)}")
