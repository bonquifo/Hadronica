# Hadronica: Standard Model Collision Laboratory

Hadronica is a desktop collision simulator for Windows. Pick beams, an energy, and a process, and it generates Standard Model events, draws them in a schematic solenoidal detector, and histograms the results. Every input comes from the 2026 Review of Particle Physics, and every formula carries its source (see the in-app Methods panel).

It has two engines:

- **Built-in:** leading-order Born cross sections with running α(s), effective couplings, and initial-state radiation. It runs anywhere Python runs.
- **Research mode (WSL):** PYTHIA 8.3 with the Monash tune, MadGraph5_aMC@NLO samples at NLO (MC@NLO, MadSpin, FxFx), Delphes detector simulation, and Rivet 4 validation against published LEP and LHC data.

**[Benchmark comparison with MadGraph, PYTHIA, Herwig 7.3 and Sherpa 3.0](https://bonquifo.github.io/Hadronica/)**

## Download

The Windows executable is attached to the [latest release](https://github.com/bonquifo/Hadronica/releases/latest). It runs the built-in engine on its own; research mode needs the WSL setup below.

## Run from source

```
pip install -r requirements.txt
python run_hadronica.py
```

Research mode needs WSL with Ubuntu. The one-time setup installs PYTHIA, Delphes, Rivet, LHAPDF and MadGraph into micromamba environments (no sudo):

```
wsl -d Ubuntu -- bash /mnt/c/<path-to-Hadronica>/hep/setup_wsl.sh
```

## Tests

```
python -m pytest
```

The suite covers the physics engine, the UI, the executable, and (when WSL is set up) the PYTHIA, MadGraph and Rivet tools. The full run takes about 12 minutes. A built executable also checks itself: `Hadronica.exe --selftest report.json`.

## Layout

| Path | Contents |
|---|---|
| `hadronica/` | The application: engine, processes, decays, detector view, Methods text |
| `hep/` | Research-mode worker, Delphes bridge, Rivet validation, MadGraph scripts, shower tune |
| `hep/compare/` | Comparisons with MadGraph, PYTHIA run directly, Herwig and Sherpa; `build_report.py` builds the page in `docs/` |
| `tests/` | Windows tests; `tests/hep/` runs inside WSL |
| `AUDIT_2026.md` | Every physics audit round, with the corrections made and their sources |
| `docs/` | The benchmark comparison page served by GitHub Pages |

## Building the executable

```
.\build_exe.ps1
```

## License

MIT; see [LICENSE](LICENSE). PYTHIA, MadGraph5_aMC@NLO, Delphes, Rivet and the other tools that research mode installs keep their own licenses.
