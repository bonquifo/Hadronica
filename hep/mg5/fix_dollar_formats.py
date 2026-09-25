"""Replace the non-standard '$' edit descriptor in MadGraph's Fortran templates.

gfortran's runtime library (libgfortran >= 15) rejects formats such as
'(a$)', which older compilers accepted as "do not advance to a new line".
The standard Fortran equivalent is advance='no'. Idempotent.

madevent_symmetry.f (used by MadSpin) also assembles such formats at run time,
write(formstr,...) '(I',nconf,'$)', then write(*,formstr); there the '$' literal
is dropped and the terminal write made non-advancing.
"""

import re
import sys

# The '$' may follow its own comma ('(i5,$)'); drop that separator too.
PATTERN = re.compile(r"write\s*\(\s*([^,()]+?)\s*,\s*'\(([^']*?)\s*,?\s*\$\s*\)'\s*\)", re.IGNORECASE)

RUNTIME = re.compile(r"write\s*\(\s*\*\s*,\s*(formstr)\s*\)", re.IGNORECASE)

DRY_RUN = "--dry-run" in sys.argv[1:]  # report what would change, write nothing

for path in [arg for arg in sys.argv[1:] if arg != "--dry-run"]:
    text = open(path, encoding="latin-1").read()
    new, count = PATTERN.subn(lambda m: f"write({m.group(1)},'({m.group(2)})',advance='no')", text)
    if "'$)'" in new:
        count += new.count("'$)'")
        new = new.replace("'$)'", "')'")
        new = RUNTIME.sub(r"write(*,\1,advance='no')", new)
    if count and not DRY_RUN:
        open(path, "w", encoding="latin-1").write(new)
    # A '$' in column 6 of fixed-form source is a continuation mark, not a format.
    leftover = [
        line.strip() for line in new.splitlines()
        if re.search(r"\$\s*\)", line) and not line.lstrip().startswith(("c", "C", "!"))
        and not (len(line) > 5 and line[5] == "$" and not line[:5].strip())
    ]
    print(f"{path}: {count} replaced, {len(leftover)} left: {leftover[:3]}")
