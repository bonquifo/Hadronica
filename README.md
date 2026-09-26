# Hadronica: Standard Model Collision Laboratory

Hadronica is a desktop collision simulator for Windows. Pick beams, an energy, and a process, and it generates Standard Model events, draws them in a schematic solenoidal detector, and histograms the results. Every input comes from the 2026 Review of Particle Physics, and every formula carries its source (see the in-app Methods panel).

It has two engines:

- **Built-in:** leading-order Born cross sections with running α(s), effective couplings, and initial-state radiation. It runs anywhere Python runs.
- **Research mode (WSL):** PYTHIA 8.3 with the Monash tune, MadGraph5_aMC@NLO samples at NLO (MC@NLO, MadSpin, FxFx), Delphes detector simulation, and Rivet 4 validation against published LEP and LHC data.

**[Benchmark comparison with MadGraph, PYTHIA, Herwig 7.3 and Sherpa 3.0](https://bonquifo.github.io/Hadronica/)**

## Download

The Windows executable is attached to the [latest release](https://github.com/bonquifo/Hadronica/releases/latest). It runs the built-in engine on its own; research mode needs the WSL setup below.

## Updates

From version 1.2.0, Hadronica checks GitHub for a newer release at most once a day, in the background. When one exists, the status bar shows **Update available**. Clicking it opens the Updates dialog: *Update and restart* downloads the new `Hadronica.exe`, checks it against the SHA-256 checksum published with the release, replaces the running program, and starts the new one. A download that does not match is discarded and nothing changes. Your saved validation results are kept.

Clicking the version number in the status bar opens the same dialog to check by hand, skip a release, or turn the daily check off. Offline, nothing happens and the app works as before. `Hadronica.exe --update [report.json]` updates without opening the window.

Versions 1.0 and 1.1 predate the updater: download 1.2.0 or later once by hand. From source, update with `git pull`.

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

## Publishing a release

Bump `__version__` in `hadronica/__init__.py`, commit and push, then:

```
python tools/release.py --notes-file notes.md
```

It builds the executable, runs its self-test, writes `Hadronica.exe.sha256`, and publishes the release with both files. Running copies only install releases that carry that checksum, and only when the tag (`v` + `__version__`) is newer than their own version.

## License

MIT; see [LICENSE](LICENSE). PYTHIA, MadGraph5_aMC@NLO, Delphes, Rivet and the other tools that research mode installs keep their own licenses.
