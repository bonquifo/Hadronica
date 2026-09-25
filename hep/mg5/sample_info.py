"""Summarize an MC@NLO LHE sample: NLO cross section, scale uncertainty, negative weights.

    python sample_info.py <sample dir> <process> <sqrt_s> <seconds> <mg5 log>

MadGraph5_aMC@NLO writes events normalized so that the cross section is the
mean event weight. The 9-point renormalization/factorization scale variation
(μ_R, μ_F ∈ {½, 1, 2} × μ₀, weight ids 1001–1009, combine='envelope') is
evaluated the same way; the uncertainty is the envelope of the nine values.
"""

import json
import re
import sys

sample, process, sqrt_s, seconds, log_path = sys.argv[1:6]
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
SCALE_IDS = [str(i) for i in range(1001, 1010)]

n = 0
negative = 0
sums = {key: 0.0 for key in ["nominal", *SCALE_IDS]}
in_event = False
first_line = False
with open(f"{sample}/events.lhe", encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        line = raw.strip()
        if line.startswith("<event"):
            in_event, first_line = True, True
            continue
        if not in_event:
            continue
        if first_line:
            first_line = False
            weight = float(line.split()[2])
            n += 1
            negative += weight < 0.0
            sums["nominal"] += weight
            continue
        if line.startswith("<wgt id="):
            match = re.match(r"<wgt id='(\d+)'>\s*([-0-9.eE+]+)", line)
            if match and match.group(1) in sums:
                sums[match.group(1)] += float(match.group(2))
        elif line.startswith("</event"):
            in_event = False

log = open(log_path, errors="replace").read()
info = {
    "process": process,
    "sqrt_s": float(sqrt_s),
    "seconds": int(seconds),
    "generator": "MadGraph5_aMC@NLO 3.5.7 (MC@NLO), NNPDF3.1 NLO",
    "events": n,
    "negative_fraction": negative / n if n else None,
}
totals = re.findall(r"Total cross section:\s*([0-9.eE+-]+)\s*\+-\s*([0-9.eE+-]+)\s*pb", log)
if totals:
    info["sigma_pb"], info["sigma_err_pb"] = float(totals[-1][0]), float(totals[-1][1])
if n:
    nominal = sums["nominal"] / n
    info["sigma_from_events_pb"] = nominal
    variations = [sums[key] / n for key in SCALE_IDS if sums[key] != 0.0]
    if len(variations) == 9 and nominal:
        info["scale_up_pct"] = 100.0 * (max(variations) / nominal - 1.0)
        info["scale_down_pct"] = 100.0 * (1.0 - min(variations) / nominal)
info.update(extra)
json.dump(info, open(f"{sample}/info.json", "w"), indent=1)
print(json.dumps(info))
