#!/usr/bin/env bash
# Produce SMLab's MC@NLO samples: 13 TeV for validation against published
# data, then 13.6 TeV (LHC Run 3) for the app. Existing samples are kept.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
while read -r proc energy events; do
    [ -z "$proc" ] && continue
    if [ -f "$HOME/smlab-cache/nlo/${proc}_${energy}/info.json" ]; then
        echo "have ${proc}_${energy}"; continue
    fi
    rm -rf "$HOME/smlab-cache/nlo/${proc}_${energy}" "$HOME/smlab-cache/nlo/${proc}_${energy}.mg5"
    echo "generating ${proc}_${energy} (${events} events)"
    bash "$HERE/generate_nlo.sh" "$proc" "$energy" "$events" | tail -1
done <<LIST
dy 13000 150000
ttbar 13000 60000
dy 13600 50000
ttbar 13600 50000
w 13600 50000
ttbar_ms 13000 60000
ttbar_ms 13600 50000
dy_fxfx 13000 150000
dy_fxfx 13600 50000
LIST
