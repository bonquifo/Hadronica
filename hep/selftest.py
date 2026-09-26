"""Drive worker.py through its JSON protocol, as the Windows bridge does.

    ~/micromamba/envs/hadronica-hep/bin/python hep/selftest.py
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
proc = subprocess.Popen(
    [sys.executable, "-u", os.path.join(HERE, "worker.py")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)


def call(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    if not reply.get("ok"):
        print("ERROR", reply.get("error"))
        print(reply.get("trace", ""))
        sys.exit(1)
    return reply


hello = call({"cmd": "hello"})
print("pythia", hello["pythia"], "processes", len(hello["processes"]))
runs = (
    ({"beams": "ee", "process": "ll_gmz_had", "sqrt_s": 91.1879, "seed": 1}, 200),
    ({"beams": "ee", "process": "ll_ww", "sqrt_s": 200.0, "seed": 2}, 200),
    ({"beams": "ee", "process": "ll_zh", "sqrt_s": 240.0, "seed": 3}, 100),
    ({"beams": "pp", "process": "pp_ttbar", "sqrt_s": 13600.0, "seed": 4}, 20),
    ({"beams": "pp", "process": "pp_z_ll", "sqrt_s": 13600.0, "seed": 5}, 50),
)
for config, n in runs:
    call({"cmd": "init", "config": config})
    reply = call({"cmd": "generate", "n": n})
    events = reply["events"]
    finals = [sum(1 for row in ev["particles"] if row[1] > 0) for ev in events]
    charged = [sum(1 for row in ev["particles"] if row[1] > 0 and abs(row[15]) > 0) for ev in events]
    displaced = sum(
        1 for ev in events for row in ev["particles"]
        if row[1] > 0 and (row[11] ** 2 + row[12] ** 2) ** 0.5 > 1.0
    )
    print(
        f"{config['process']:12s} √s={config['sqrt_s']:>8}  σ={reply['sigma_pb']:.4g} ± {reply['sigma_err_pb']:.2g} pb"
        f"  <final>={sum(finals) / n:.1f}  <charged>={sum(charged) / n:.1f}"
        f"  displaced>1mm/ev={displaced / n:.2f}  jets(ev0)={len(events[0]['jets'])}  {reply['seconds']:.2f}s"
    )
call({"cmd": "quit"})
