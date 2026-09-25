"""Undo MadSpin's uniform branching-ratio factor on the weights of a decayed LHE file.

    python restore_weights.py <undecayed events.lhe> <decayed events.lhe>

MadSpin's on-shell mode multiplies every event weight by (Γ_decay / Γ_card) per
decayed top, where Γ_decay is its own t → b f f' width with a finite-width W and
Γ_card the parameter-card width (narrow W). With every decay channel kept the
branching fraction is 1, so the factor is a pure normalization artifact (0.9538
for t t~). This script checks that the event-by-event ratio really is one
constant, then divides it out of every weight, including the scale variations,
so the file again reproduces the NLO cross section. Prints the factor.
"""

import re
import sys

WGT = re.compile(r"(<wgt id=['\"][^'\"]+['\"]>\s*)([-+0-9.eE]+)(\s*</wgt>)")


def event_weights(path: str) -> list[float]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            if line.strip().startswith("<event"):
                out.append(float(next(lines).split()[2]))
    return out


def main() -> None:
    undecayed, decayed = sys.argv[1], sys.argv[2]
    before, after = event_weights(undecayed), event_weights(decayed)
    if len(before) != len(after):
        sys.exit(f"event counts differ: {len(before)} vs {len(after)}")
    ratios = [b / a for a, b in zip(before, after) if a]
    factor = sum(ratios) / len(ratios)
    spread = max(abs(r / factor - 1.0) for r in ratios)
    if spread > 1e-5:
        sys.exit(f"weights were not scaled uniformly (spread {spread:.2e}); not rescaling")
    out_lines = []
    with open(decayed, encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        for line in lines:
            out_lines.append(line)
            if line.strip().startswith("<event"):
                header = next(lines)
                fields = header.split()
                fields[2] = f"{float(fields[2]) / factor:+.10e}"
                out_lines.append(" " + " ".join(fields) + "\n")
                for body in lines:
                    if "<wgt" in body:
                        body = WGT.sub(lambda m: f"{m.group(1)}{float(m.group(2)) / factor:+.10e}{m.group(3)}", body)
                    out_lines.append(body)
                    if body.strip().startswith("</event"):
                        break
    with open(decayed, "w", encoding="utf-8") as handle:
        handle.writelines(out_lines)
    print(f"{factor:.6f}")


if __name__ == "__main__":
    main()
