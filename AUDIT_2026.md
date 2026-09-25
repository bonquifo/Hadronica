# SMLab physics audit and UI rebuild (September 2026)

> **Renamed.** On 25 September 2026 the application was renamed **Hadronica** (still subtitled Standard Model Collision Laboratory). The rounds below keep the name it had when they were written: "SMLab" is Hadronica, and the "SMLab-AZ 2026" shower tune is now "Hadronica-AZ 2026".

The original Grok-built code is preserved in `_original_backup/`.

## Inputs updated to current sources

The app cited the 2024 Review of Particle Physics. The 2026 edition is published, so every input now comes from it, with CODATA 2022 for the fundamental constants:

- F. Takahashi et al. (Particle Data Group), Int. J. Mod. Phys. A 41, 2630011 (2026).
- CODATA 2022.

Each value in `smlab/constants.py` carries its source.

| Quantity | Was | Now |
|---|---|---|
| (ħc)² in pb·GeV² | 3.89379292e8 (wrong digits) | 3.893793721e8 (exact) |
| α⁻¹(0) | 137.035999084 (CODATA 2018) | 137.035999177 (CODATA 2022) |
| G_F | 1.1663788e-5 | 1.1663785e-5 GeV⁻² (PDG 2026; CODATA 2022 lists 1.1663787e-5) |
| sin²θ_eff | 0.23153 | 0.23154 |
| M_W, Γ_W | 80.3692, 2.085 | 80.3625, 2.14 GeV |
| M_Z | 91.1880 | 91.1879 GeV |
| M_H | 125.20 | 125.13 GeV |
| m_t | 172.57 | 172.60 GeV |
| m_s, m_c, m_b | 93.5 MeV, 1.273, 4.183 | 92.9 MeV, 1.2729, 4.186 GeV |
| CKM \|V_ud\|, \|V_cd\|, \|V_cs\|, \|V_cb\|, \|V_ub\| | 2022 values | 0.97367, 0.2247, 0.969, 0.0407, 0.00389 |
| Higgs BRs, Γ_H | YR4 at 125.0 GeV, 4.07 MeV | YR4 at 125.10 GeV, 4.101 MeV |
| τ → eνν, μνν | 17.82 %, 17.39 % | 17.85 %, 17.37 % |
| ρ(770) | one mass/width for ρ⁰ and ρ± | separate ρ⁰ and ρ± values |

## Physics errors fixed

1. **Improved Born approximation used α(0).** The "improved Born" formula needs the running α(s), and using α(0) made the photon exchange about 13 % low at LEP energies. The app now uses α(s), with the hadronic part from the PDG Δα_had⁽⁵⁾(M_Z²) = 0.02760. This gives α(M_Z)⁻¹ = 128.95.
2. **Forward–backward term for massive fermions** was scaled by β; it must be scaled by β². This overstated the t t̄ asymmetry near threshold by 1/β.
3. **α_s running** used the wrong number of flavors between M_Z and each threshold, and was discontinuous at the thresholds. It is now one-loop with continuous matching at m_c, m_b and m_t.
4. **e⁺e⁻ → ν_e ν̄_e ignored t-channel W exchange.** The W contribution is now included, and it dominates above the Z (for example 40 pb against 1 pb at 200 GeV). The same applies to μ⁺μ⁻ → ν_μ ν̄_μ.
5. **Initial-state radiation.** The radiator lacked its O(β) terms, and the 640-point grid put about one point across the Z peak. Radiative-return rates came out about 11 % low (6.75 pb against 7.55 pb for μμ at 200 GeV). The app now uses the complete O(β) Kuraev–Fadin radiator on an adaptive grid that converges to better than 0.02 %.
6. **H → WW\*/ZZ\*** virtual masses were sampled without the q² Jacobian (m₁m₂), which biased the off-shell mass low.
7. **τ decays:** the 4.62 % channel (π⁻π⁺π⁻π⁰ν, three charged tracks) was generated as a one-prong π⁻3π⁰.
8. **W partial widths** now use the exact tree-level V−A mass factor.
9. **Angular sampling** snapped cosθ to grid points; the inverse CDF is now continuous.
10. **B-field slider** could reach 16 T instead of stopping at 4 T.

## Verification against data

- **μμ pole cross section:** 1.98 nb here, against the LEP pole fit σ⁰_had/R_μ = 41.480/20.784 = 2.00 nb (PDG 2026 Electroweak review, Table 10.3).
- **Z → ℓℓ branching fraction:** 3.36 % here, against 3.363 % (e⁺e⁻) in PDG 2026.
- **W → ℓν branching fraction:** 10.88 % here, against 10.86 ± 0.09 % in PDG 2026.
- **ZH at 250 GeV:** 238 fb at Born level. The SANC reference gives 225.6 fb (Born, α(0) scheme; S. Bondarenko et al., Phys. Rev. D 100, 073002 (2019)). Converting its couplings to the G_F scheme raises that by about 7 %, to about 241 fb.
- **Sources checked:** the Kilian–Krämer–Zerwas and Djouadi formula references, and the CMS solenoid parameters (arXiv:2201.07557), were checked against the original papers.

## What remains approximate

Everything here is leading order. The "NOT INCLUDED" section of Methods lists what is left out, for example showers, hadronization, PDFs, final-state radiation, full one-loop EW corrections, threshold QCD for t t̄, and W⁺W⁻ production.

## Tests

`python -m pytest` runs 81 tests. 14 are new; they cover each fix above and the new UI layout.

# Round 2: research-grade engines (September 2026)

A **PYTHIA 8.3** switch in the top bar replaces the built-in leading-order generator with the programs used in LHC and future-collider studies. They run in WSL (Ubuntu), in a conda-forge environment at `~/micromamba/envs/smlab-hep`.

| Component | Version | Role | Reference |
|---|---|---|---|
| PYTHIA | 8.312 | Hard process, QED/QCD showers, multiparton interactions, Lund hadronization, decays | C. Bierlich et al., SciPost Phys. Codebases 8 (2022), arXiv:2203.11601 |
| Monash 2013 tune | PYTHIA default | Shower, MPI and hadronization parameters | P. Skands, S. Carrazza, J. Rojo, Eur. Phys. J. C 74 (2014) 3024, arXiv:1404.5630 |
| NNPDF2.3 QCD+QED LO | built into PYTHIA | Proton PDFs of the Monash tune; LHAPDF 6.5.6 and NNPDF3.1 NNLO are also installed | R. D. Ball et al., Nucl. Phys. B 877 (2013) 290 |
| FastJet | 3.5.1 | Anti-kT R = 0.4 jets for pp, Durham jets for e⁺e⁻ | M. Cacciari, G. P. Salam, G. Soyez, Eur. Phys. J. C 72 (2012) 1896; anti-kT: JHEP 04 (2008) 063 |
| HepMC3 | 3.3 | Event record handed to the detector simulation | A. Buckley et al., Comput. Phys. Commun. 260 (2021) 107310 |
| Delphes | 3.5.1 | Detector response: CMS (pp), ALEPH (LEP), IDEA (FCC-ee), CLICdet, muon-collider detector | J. de Favereau et al., JHEP 02 (2014) 057, arXiv:1307.6346 |

**Setup on a new machine.** Run this once:

```
wsl -d Ubuntu -- bash /mnt/c/ParticleCollision/hep/setup_wsl.sh
```

It installs micromamba and the environment. It also fetches two Delphes card include folders that the conda-forge package omits.

**How it fits together.**
- `hep/worker.py` is the WSL-side worker; it speaks JSON lines.
- `hep/detector.py` converts events to HepMC3 and runs Delphes.
- `smlab/engine.py` is the Windows-side bridge; it runs on a background thread.
- `smlab/app_pythia.py` holds the PYTHIA panels.
- `smlab/fullscene.py` draws full events: tracks from their true production vertices, displaced K⁰_S/Λ/heavy-flavour decays, calorimeter towers, jets, and missing momentum.

**Checks.**

| Check | Result | Reference |
|---|---|---|
| σ(e⁺e⁻ → Z → hadrons) at the pole, with ISR | 29.6 ± 1.2 nb | ≈ 30.5 nb: σ⁰_had = 41.480 nb (PDG 2026) lowered about 25 % by initial-state radiation |
| σ(e⁺e⁻ → W⁺W⁻) at 200 GeV | 16.5 pb | LEP measured 16.77 ± 0.29 pb at 199.5 GeV; RacoonWW/YFSWW predict 17.0 pb (Phys. Rept. 532 (2013) 119) |
| Energy, momentum, and charge in each event | conserved to about 10⁻⁷ GeV | — |
| Delphes CMS muon reconstruction in acceptance | 87 % | — |

**Tests.** `tests/test_engine.py` contains unit tests and an integration test against the real WSL engine. The integration test skips itself when the engine is not installed.

**Limits that remain.**
- **Order:** PYTHIA's hard processes are leading order. Rates can sit below NNLO predictions; for example, 602 pb against 923.6 pb for t t̄ at 13.6 TeV (NNLO+NNLL, LHC Top Working Group, computed with Top++: M. Czakon, A. Mitov, Comput. Phys. Commun. 185 (2014) 2930). ATLAS measured 850 ± 27 pb (Phys. Lett. B 848 (2024) 138376) and CMS 881 ± 30 pb (JHEP 08 (2023) 204).
- **Detector:** Delphes is a parametrized fast simulation, not Geant4.
- **Not modelled:** pileup, beam polarization, and beamstrahlung.

# Round 3: validation, pileup, and NLO (September 2026)

The three additions were built in this order:
1. **Rivet**, first, so each later step is checked against published data.
2. **Pileup.**
3. **MadGraph5_aMC@NLO.**

## Validation against data (Rivet 4.1.4)

Run `hep/validate.py` or use **Validation** in the app. It generates each benchmark in parallel on all cores, runs the Rivet analysis, and scores every distribution with χ²/ndf. It reports shape and rate (MC/data) separately. Total data uncertainties are used; per-plot titles and axis labels come from Rivet's `.plot` files.

| Benchmark (Rivet analysis; measurement) | LO + shower χ²/ndf | MC@NLO + shower χ²/ndf |
|---|---|---|
| LEP Z → hadrons, event shapes (ALEPH_1996_I428072; ALEPH, Phys. Rept. 294 (1998) 1) | 2.5 (rate 0.98) | — |
| LHC 13 TeV minimum bias (ATLAS_2016_I1419652; ATLAS, Phys. Lett. B 758 (2016) 67) | 36 (shape 12.6, rate 1.09) | — |
| LHC 13 TeV Z pT (ATLAS_2019_I1768911; ATLAS, Eur. Phys. J. C 80 (2020) 616) | 4.2 | 25.8 |
| LHC 13 TeV t t̄ event variables (CMS_2018_I1662081; CMS, JHEP 06 (2018) 002) | 7.4 (shape 1.6) | **2.2** (shape 1.3) |
| LHC 13 TeV inclusive jets (CMS_2016_I1459051; CMS, Eur. Phys. J. C 76 (2016) 451) | 5.7 (rate 1.03) | — |

What the results show:

- **Minimum bias.** The Monash tune overshoots charged-particle production at 13 TeV by about 9 %. This is a limitation of the tune.
- **t t̄.** NLO improves agreement clearly.
- **Z pT is not a matching error.** Above 100 GeV the showered and unshowered MC@NLO tails agree to within 10 % (the highest bin identically), so the shower is correctly matched. The high-pT tail of an inclusive NLO Z sample is only LO Z + 1 jet, and the peak depends on MadGraph's untuned shower settings. The fix would be NLO merging (FxFx).

**Bugs found in my own first comparison code, before any result was trusted:**

1. **MC errors silently set to 0.** Rivet 4 names its uncertainty sources, and the default source is empty.
2. **Reference errors also set to 0 for three analyses.** Those analyses produced no plots.
3. **Weights written through pyHepMC3 were silently dropped.** They are now written with pyhepmc, and the sign is verified.

## Pileup (Delphes CMS pileup card)

- **How it works.** Minimum-bias collisions are overlaid on each event: a Poisson number with mean μ = 30, 60, or 140, drawn from a cached library of 5000 PYTHIA events, with the card's beam-spot profile.
- **Display.** Pileup tracks are flagged and drawn from their own vertices.
- **Checks.** At μ = 60 the reconstruction finds 60–68 vertices. The mean missing pT in Z → ℓℓ events rises from 17 GeV (μ = 0) to 36 GeV (μ = 30) and 50 GeV (μ = 60).

## NLO (MadGraph5_aMC@NLO 3.5.7, MC@NLO, NNPDF3.1 NLO)

**Setup.**
- **Generator.** MadGraph5_aMC@NLO (J. Alwall et al., JHEP 07 (2014) 079, arXiv:1405.0301).
- **Inputs.** PDG 2026, with α⁻¹ = 132.04 so that the G_F scheme gives M_W = 80.3617 GeV.
- **Shower matching.** MC@NLO (S. Frixione, B. Webber, JHEP 06 (2002) 029), showered by PYTHIA with MadGraph's own PYTHIA 8 MC@NLO settings.
- **Pole check.** The virtual corrections pass MadGraph's pole-cancellation check.

**Samples.**

| Sample | σ(NLO) | Scale uncertainty | Negative weights |
|---|---|---|---|
| Z/γ* → ℓℓ, 13 TeV | 3.69 nb | +5.5/−10.3 % | 7 % |
| t t̄, 13 TeV | 663 pb | +9.2/−10.6 % | 23 % |
| Z/γ* → ℓℓ, 13.6 TeV | 3.87 nb | +5.6/−10.4 % | 7 % |
| t t̄, 13.6 TeV | 737 pb | +9.7/−10.7 % | 23 % |
| W → ℓν, 13.6 TeV | 38.9 nb | +6.7/−11.7 % | 7 % |

**Making the conda build work needed several toolchain fixes, all scripted in `hep/mg5/setup_mg5.sh`:**
- FastJet 3.5.1 built from source, because the conda package lacks the headers.
- gcc/gfortran 13.
- OneLOop, Ninja, and Collier rebuilt.
- `-lavh_olo` added after the conda IREGI in the link line.
- The non-standard `'(…$)'` Fortran formats that newer libgfortran rejects rewritten as `advance='no'`.

**Regenerating samples.** Run `hep/mg5/produce_all.sh`.

## Tests

`python -m pytest` ran 93 tests at the end of Round 3. They include the Validation panel, signed-weight histograms, pileup tracks from their own vertices, and a live WSL test with pileup.

# Round 4: MadSpin, FxFx merging, and a shower tune (September 2026)

These three steps address what Round 3 left open:
1. **MadSpin top decays**, so t t̄ keeps its spin correlations.
2. **FxFx NLO merging** for Z + jets, so the Z transverse-momentum tail is NLO accurate.
3. **A re-tune of the non-perturbative shower parameters**, fitted to the low-pT Z data and checked on t t̄ data not used in the fit.

## MadSpin (Artoisenet, Frederix, Mattelaer, Rietkerk, JHEP 03 (2013) 015)

**What it does.** The top quarks of the MC@NLO t t̄ samples are decayed by MadSpin with the full tree-level t → W b → f f′ b matrix element. This keeps the t–t̄ spin correlation and the off-shell line shape (Breit–Wigner, cut at 15 Γ). The decays are run standalone on the undecayed NLO events by `hep/mg5/decay_madspin.sh`. The NLO cross section is unchanged, because every decay channel is kept.

**A fix it needed.** MadSpin failed at first. It compiles `madevent_symmetry.f`, which assembles a `'(I…$)'` Fortran format at run time, and that is a form the Round 3 fixer could not see. `fix_dollar_formats.py` now handles it, and `setup_mg5.sh` also scans `madgraph/`.

**Results** (superseded: Round 5 found that MadSpin's off-shell mode biased the top line shape and that the Δφ comparison below is limited by MC statistics; see Round 5 for the corrected samples and numbers).

| Benchmark | NLO, PYTHIA decays | NLO + MadSpin |
|---|---|---|
| CMS t t̄ event variables, median χ²/ndf | 2.22 | **1.29** |
| ATLAS eμ Δφ(e, μ) shape χ²/ndf (Eur. Phys. J. C 80 (2020) 528) | 1.94 (3.0 at LO) | **0.87** |

Δφ(e, μ) is the classic t t̄ spin-correlation observable, but at this sample size the MC statistical error (15–30 % per bin) is five times the data error, so the change from 1.94 to 0.87 is not significant (retracted in Round 5).

**Caveat.** MadSpin's branching ratios are tree level: BR(W → ℓν) is 1/9 against the measured 10.86 %, so dilepton rates are about 5 % high.

## FxFx merging (Frederix, Frixione, JHEP 12 (2012) 061)

**The sample.** Z/γ* → ℓℓ + 0, 1, 2 jets, each at NLO, at 13 TeV:
- 150 000 events
- merging scale Qcut = 20 GeV, ptj = 10 GeV
- 29 % negative weights before the veto, 21 % after

**A fix it needed.** PYTHIA's `JetMatching:*` settings act only through the `JetMatchingMadgraph` user hook, and the Python bindings do not expose it. Without the hook the settings are silently ignored and the sample is showered unmerged: σ came out at 6.66 nb with no vetoes.

`hep/ext/smlab_fxfx.cpp` is a 30-line pybind11 module, built by `hep/ext/build_fxfx.sh` and called from `setup_wsl.sh`. It attaches PYTHIA's own `CombineMatchingInput` hook to the Python `Pythia` object, exactly as PYTHIA's FxFx example program does. The module shares pybind11 internals v12 with `pythia8.so`.

**Safeguards.**
- If the module is missing, the worker hides FxFx samples rather than showering them unmerged.
- **Chunk spacing.** About 56 % of LHE events are vetoed (4502 read per 2000 accepted). Validation chunks therefore start at evenly spaced positions in the LHE file, and at most 40 % of the file is used, so no event is showered twice.
- **Displayed σ.** For FxFx, MadGraph's σ is before merging, so the app and validation use PYTHIA's merged σ after the veto: 4.20 nb for m_ℓℓ > 60 GeV, e + μ (60 000 events; 4.19 nb with the tune, since the veto depends on the shower).

## Shower tune SMLab-AZ 2026 (method: ATLAS AZNLO, JHEP 09 (2014) 145)

**What is fitted.** `hep/tune.py` scans `BeamRemnants:primordialKThard` × `SpaceShower:pT0Ref` on the FxFx sample:
- 60 000 showered events per point
- the same LHE events at every point
- the shower α_s kept at 0.118 for consistent matching
- scored on the shape of ATLAS_2019_I1768911 for pT(ℓℓ) < 30 GeV and φ*_η < 0.3 (38 bins)

**Scan history.**
1. **First 5 × 4 grid:** the best point sat in a corner.
2. **Extended grid:** χ² falls towards pT0Ref's hard lower limit (0.5 GeV; ATLAS's AZ tune found 0.59).
3. **Profile fit:** the 2D quadratic has no interior minimum, because the surface is noisy along pT0Ref at the ±15 level in χ². pT0Ref was therefore fixed at 0.5 and kT fitted along that row.

**Result.**
- primordialKThard = **3.04 ± 0.17 GeV** (Δχ² = 1); pT0Ref = 0.5 GeV.
- χ² drops from 259 with the Monash values to about 45 for 38 bins.
- `tune.py --refit` reproduces the fit from `hep/validation/tune.json`.

**Interpretation.** The 3 GeV kT is an effective parameter. It also absorbs soft radiation missing from a shower run with MadGraph's MC@NLO settings, which use one-loop α_s = 0.118 and no CMW scheme.

**Where it applies.** The worker applies the tune to every NLO sample. The app has a switch for it, "SMLab-AZ 2026 shower tune", on by default. It is never applied at leading order, where Monash applies.

Superseded for t t̄ by Round 5 (corrected widths and on-shell MadSpin decays); the Z rows are unchanged.

| Benchmark | χ²/ndf |
|---|---|
| ATLAS Z pT(ℓℓ), full spectrum: LO / MC@NLO / FxFx / **FxFx + tune** | 4.23 / 25.8 / 6.38 / **1.65** |
| ATLAS φ*_η: LO / MC@NLO / FxFx / **FxFx + tune** | 3.15 / 15.9 / 2.91 / **1.57** |
| CMS t t̄ event variables, MadSpin / **+ tune** (not used in the fit) | 1.29 / **1.24** |
| ATLAS eμ t t̄, MadSpin / **+ tune** (not used in the fit) | 1.37 / **1.24** (shape 1.04 → 0.71) |

The Z pT result is not an independent test, because its low-pT part was fitted. The t t̄ results are independent.

**A caveat about the scoring.** The highest-m(eμ) slices of the dilepton measurement contain only a few dozen of the 60 000 MC events. Empty MC bins there give χ²/ndf of 10–30 for every generator. The Validation panel now marks such plots with * as limited by MC statistics.

## Samples

| Sample | σ | Notes |
|---|---|---|
| t t̄ + MadSpin, 13 TeV | 663 pb | 60 000 events (redecayed on-shell in Round 5) |
| t t̄ + MadSpin, 13.6 TeV | 737 pb | 50 000 events (redecayed on-shell in Round 5) |
| Z + 0, 1, 2 j FxFx, 13 TeV | 4.20 nb after merging | 150 000 LHE events |
| Z + 0, 1, 2 j FxFx, 13.6 TeV | 4.46 nb after merging | 50 000 LHE events; the app's default for Z at 13.6 TeV |

## Tests

`python -m pytest` covers:
- the fit-region χ²
- the 2D quadratic and 1D profile fits
- the app requesting the tune only for NLO samples it applies to
- the flag for plots limited by MC statistics

# Round 5: an objective test suite, and what it found (September 2026)

The research-grade stack from Rounds 2–4 had almost no tests. The new tests compare against things outside the code: exact expectations built into synthetic inputs, the PDG 2026 values in `smlab/constants.py`, independent physics references, and independent recomputations of stored results.

**How it runs.** `python -m pytest` runs both halves:
- **Windows side:** `tests/test_hep_tools.py` plus the app tests.
- **WSL side:** `tests/hep/`, which uses the real PYTHIA, pyhepmc, gfortran and MadGraph samples. `tests/test_wsl_suite.py` runs it from Windows.

**What they found.** Writing them found seven real problems, all now fixed:

1. **PYTHIA ignored the PDG widths.** It recomputes resonance widths from its own partial widths when it starts, which gave Γ_W = 2.092 GeV and Γ_t = 1.347 GeV (5 % low).
   - **Fix:** `doForceWidth` now keeps the PDG totals, with the partial widths rescaled so branching ratios are unchanged.
   - **Z exception:** PYTHIA does not honour the flag for the Z, because its γ*/Z treatment derives the width from the couplings. That gives 2.504 GeV, 0.34 % above PDG, and the Z-pole hadronic cross section is 41.46 nb against the measured 41.48 nb. A test checks both.
2. **MadSpin's default off-shell mode biased the top line shape.** Tops peaked at 173.03 GeV instead of 172.60, with twice as many above the pole as below within one width.
   - **Cross-check:** MadGraph's full leading-order t t̄ → bW bW matrix element with identical inputs gives a symmetric Breit–Wigner (median 172.62 GeV; 3589 vs 3606 tops within one width below and above).
   - **Fix:** the samples are now decayed in MadSpin's on-shell mode, which keeps the full spin correlations. Tops are exactly at 172.6 GeV, and the W decays keep their Breit–Wigner. The 1.4 GeV top width that this drops is far below detector resolution.
   - **Two follow-on fixes:**
     - On-shell mode compiles with f2py, which failed with setuptools ≥ 65; `SETUPTOOLS_USE_DISTUTILS=stdlib` fixes it.
     - On-shell mode also multiplies every weight by a constant 0.9538, its own decay width over the parameter-card width. `restore_weights.py` checks the factor is uniform event by event and divides it out, so the events again reproduce the NLO cross section.
   - **Radiative decays.** About 1.1 % of tops (1361 of 120 000) decay radiatively, t → b W γ. MadSpin leaves those W's for PYTHIA to decay, without the spin correlation. That dilutes the correlation by at most about 2 %, well inside the ±0.039 error on D. The tests conserve four-momentum over all top daughters, photon included.
   - **Measured correlation:** D = −0.244 ± 0.039 over 6504 dileptonic events.
3. **The Δφ(e, μ) spin-correlation claim of Round 4 was not significant.** Only about 1 in 60 t t̄ events passes the ATLAS eμ selection. The MC statistical error per Δφ bin is 15–30 %, against 3 % for the data, and the correlated and uncorrelated samples differ by χ²/ndf 0.93, i.e. not at all.
   - **Where the spin correlation is shown instead:** a test on the LHE file measures the opening-angle coefficient D = −3⟨cos φ⟩ with signed weights. It is compared with the Standard Model prediction of −0.24 at 13 TeV (NLO QCD with weak corrections, from Table 7 and Eq. 4.23 of Bernreuther, Heisler, Si, JHEP 12 (2015) 026); uncorrelated decays give 0.
   - **How the panel marks such plots:** it flags a plot as MC-statistics limited when the simulation's error exceeds the data's in most bins. A χ²/ndf near 1 there means consistent within MC statistics, not agreement at the data's precision.
   - **Scale of the problem:** under that criterion, all 40 dilepton plots, both Z pT plots, and most t t̄ and jet plots are MC limited. Large χ² values, such as minimum bias (36) and jets (5.7), remain real disagreements.
4. **The Fortran fixer turned `'(i5,$)'` into the invalid `'(i5,)'`.** A test now compiles and runs the fixed code with MadGraph's own gfortran and libgfortran. The original aborts ("Missing comma"); the fixed program prints the expected single line.
5. **`setup_mg5.sh` had Windows line endings** from an edit earlier in this session, and a fresh WSL setup would have failed. A test now parses every shell script and rejects CRLF.
6. **The docs quoted the merged FxFx σ as 4.19 nb**, but the stored untuned result is 4.20 nb. A test now checks every number quoted here against `results.json` and `tune.json`.
7. **Two test tolerances were my own guesses, both wrong.** The momentum-balance test now uses PYTHIA's documented `Check:epTolWarn` (10⁻⁶ of √s). The MadSpin line-shape test uses the MadEvent reference above.

## Corrected results (all benchmarks rerun with the width fix and the on-shell t t̄ samples)

| Benchmark | LO | MC@NLO | NLO + MadSpin | FxFx | + SMLab tune |
|---|---|---|---|---|---|
| CMS t t̄ event variables | 7.92 | 2.51 | 2.01 | — | 1.82 |
| ATLAS eμ t t̄ (all 40 plots MC limited) | 11.34 | 1.50 | 0.97 | — | 1.19 |
| ATLAS Z pT(ℓℓ) (MC limited) | 4.23 | 25.8 | — | 6.38 | 1.65 |
| ATLAS φ*_η (MC limited) | 3.15 | 15.9 | — | 2.91 | 1.57 |
| ALEPH event shapes | 2.48 | — | — | — | — |
| ATLAS minimum bias | 36.3 | — | — | — | — |
| CMS inclusive jets | 5.68 | — | — | — | — |

The LEP, minimum-bias, jet and Z results are identical to Round 4. None of those processes depends on the W or top width, and PYTHIA does not force the Z width.

## What the tests check

**Windows side, no WSL needed:**
- the Fortran fixer
- the LHE summary (σ, negative weights, scale envelope) and the MadSpin weight restoration, both on synthetic samples with known answers
- that validation chunks never reuse an LHE event, including under the measured FxFx veto rate
- that every stored χ², shape χ², rate and median follows from the stored histograms
- that the tune is reproduced from its stored scan and lies within PYTHIA's limits
- that PYTHIA and MadGraph use the PDG 2026 inputs, and that α⁻¹ = 132.04 gives M_W = 80.3617 GeV in the G_F scheme
- that every number quoted in this file matches the results
- that every file the exe needs is bundled

**WSL side:**
- PYTHIA's effective widths and the LEP Z-pole cross section
- four-momentum conservation
- where the tune is applied
- the HepMC3 round trip with negative weights
- that without the FxFx hook nothing is vetoed, and with it the veto acceptance and merged σ are consistent with the inclusive NLO σ and scale with energy like it
- MadSpin samples:
  - σ and events preserved
  - four-momentum conservation in every event
  - t and W line shapes
  - BR(W → ℓν) = 1/9 per flavour
  - the spin-correlation coefficient D
- compiling and running the fixed Fortran
- shell-script syntax

# Round 6: testing the whole application (September 2026)

The earlier rounds test components. This round tests the application the way a user drives it.

## The shipped executable tests itself

`SMLab.exe --selftest [report.json]` runs the real application headless and writes a JSON report; the exit code is 0 only if every check passes. It covers:
- **The built-in engine:** every beam, preset and process, 636 events.
- **The PYTHIA engine**, started through the GUI's own bridge from the files bundled in the exe:
  - e⁺e⁻ → Z → hadrons at the Z pole
  - e⁺e⁻ → ZH at 240 GeV with the IDEA detector
  - μ⁺μ⁻ → t t̄ at 3 TeV with the muon-collider detector
  - pp → H → 4ℓ at 13.6 TeV
  - pp → Z at NLO (FxFx) with the SMLab tune
  - pp → t t̄ at NLO + MadSpin with the CMS detector and pileup μ = 60
- **Every event** must conserve four-momentum and charge, and every panel and view is drawn.

**Result.** The frozen build passes, in 48 s, and leaves no worker process running in WSL.

## System tests (`tests/test_system.py`)

- **Built-in engine sweep.** 17 energies (range edges, the Z pole and flanks, and just above each threshold) × every beam × ISR on and off × every allowed process: more than 1500 events, all conserving. Every cross section shown is finite and non-negative.
- **Custom collisions.** Every species pair the picker offers (38 × 38), at random momenta and angles.
  - √s must equal the invariant mass computed independently from the chosen momenta and opening angle.
  - A pair must be supported exactly when it is the particle–antiparticle pair of a modelled beam.
  - Unsupported pairs show no fabricated event.
- **Random sessions on the real interface.**
  - 2500 clicks, drags and scrolls on the built-in engine, and 400 on the live PYTHIA engine.
  - Nothing may raise, every displayed event must conserve, and every PYTHIA request must be answered.
- **Every PYTHIA process at every preset energy**, for e⁺e⁻, μ⁺μ⁻ and pp: about 150 configurations through the GUI's bridge. Every event conserves, and every σ is positive and finite.
- **PYTHIA cross sections against references fixed in advance** (`tests/hep/test_cross_sections.py`, 2500 events each):

| Process | SMLab (PYTHIA) | Reference |
|---|---|---|
| e⁺e⁻ → Z → hadrons, peak with ISR | 30.29 ± 0.30 nb | ≈ 30.5 nb: σ⁰_had = 41.480 nb (PDG 2026) lowered about 25 % by initial-state radiation |
| e⁺e⁻ → W⁺W⁻, 200 GeV | 17.33 ± 0.15 pb | 17.0 pb predicted (RacoonWW/YFSWW); LEP measured 16.77 ± 0.29 pb at 199.5 GeV |
| e⁺e⁻ → ZZ, 200 GeV | 0.997 ± 0.007 pb | 0.98 pb predicted (YFSZZ/ZZTO, NC02); LEP measured 0.95 ± 0.12 pb at 199.5 GeV |
| e⁺e⁻ → ZH, 240 GeV | 0.1999 ± 0.0017 pb | 196.2 fb (CEPC CDR Vol. II, Table 11.2, arXiv:1811.10545) |

## What it found

**The "Z Z" processes included photon pairs.** `ll_zz` and `pp_vv` used PYTHIA's `ffbar2gmZgmZ`, whose default `WeakZ0:gmZmode = 0` adds γ*γ* and γ*Z pairs down to 20 GeV. That gave 1.35 pb at 200 GeV, 36 % above the LEP measurement.
- **Fix:** `WeakZ0:gmZmode = 2`, pure Z pairs.
- **Result:** 0.997 pb.
- **Validation benchmarks:** none of them uses these processes.

Everything else behaved correctly under these tests.

# Round 7: comparison with the established generators (September 2026)

SMLab's research mode is itself built on PYTHIA 8, MadGraph5_aMC@NLO, Delphes and Rivet. The comparison therefore has three parts, all reproducible with the scripts in `hep/compare/`.

**References for the generators compared.**
- MadGraph5_aMC@NLO: J. Alwall et al., JHEP 07 (2014) 079, arXiv:1405.0301.
- PYTHIA 8.3: C. Bierlich et al., SciPost Phys. Codebases 8 (2022), arXiv:2203.11601.
- Herwig 7.3: G. Bewick et al., Eur. Phys. J. C 84 (2024) 1053, arXiv:2312.05175.
- Sherpa 3: Sherpa Collaboration, E. Bothmann et al., JHEP 12 (2024) 156, arXiv:2410.22148.
- Rivet 4: C. Bierlich et al., SciPost Phys. Codebases 36 (2024), arXiv:2404.15984; YODA 2: A. Buckley et al., SciPost Phys. Codebases 45 (2025), arXiv:2312.15070.

## 1. The built-in engine against MadGraph5_aMC@NLO

`builtin_vs_madgraph.py` compares SMLab's own Born engine with MadGraph at leading order, for 11 processes at 25 energy points, with ISR off. The processes are μμ, ττ, uū, dd̄, bb̄, tt̄, ν_μν̄_μ, ν_eν̄_e (with W exchange), ZH, Bhabha and γγ.

**Implementation check.** With SMLab's couplings switched to MadGraph's scheme (α = 1/132.04, on-shell sin²θ_W, no QCD factor):
- every cross section agrees to within **0.13 %**;
- the forward–backward asymmetries of e⁺e⁻ → μ⁺μ⁻ agree within MadGraph's statistical errors (pulls −1.4, −0.7, −0.2, +0.9).

This verifies the formulas: the γ/Z interference, the massive-fermion thresholds, the t-channel W, ZH, and the QED processes with the angular cut.

**As shipped.** SMLab's improved Born choices differ from MadGraph's tree level by a few percent, all explained:

| Difference | Cause |
|---|---|
| +5.6 to +7.3 % for μμ at 250–500 GeV | The running α(s) |
| +1.6 to +2.1 % for quarks at the Z pole | The QCD factor and sin²θ_eff |
| −0.6 % for ZH | sin²θ_eff |

At the Z pole these choices give A_FB(μμ) = 0.0161, close to LEP's measured 0.0169 ± 0.0013 (A_FB^(0,μ), PDG 2026 Electroweak review, Table 10.3, whose SM prediction is 0.01618 ± 0.00006). MadGraph's tree-level scheme gives 0.038.

## 2. SMLab's PYTHIA mode against PYTHIA run directly

`pythia_direct.py` and `pythia_direct.cc` feed the settings from SMLab's worker to a stand-alone C++ PYTHIA program. That program converts events with PYTHIA's own HepMC3 interface and analyses them with a Rivet 4 handler, bypassing SMLab's bridge, serialization, HepMC conversion and weight handling.

**Identical events.** For the same events (same seed, 20 000 minimum-bias events), SMLab's path and PYTHIA's path give **bit-identical** Rivet histograms.

**Independent seeds.** Over the five leading-order benchmarks, the MC-to-MC median χ²/ndf is 1.02 (LEP), 1.74 (minimum bias), 1.00 (Z pT), 0.70 (t t̄) and 1.01 (jets).
- The minimum-bias value reflects strongly correlated bins: a direct-against-direct control with two different seeds gives 1.14 on the same metric.
- With the identical-events result above, that shows SMLab's layers change nothing.

## 3. Against Herwig 7.3 and Sherpa 3.0

`other_generators.py` runs the six validation benchmarks with Herwig 7.3.0 (built from source with herwig-bootstrap) and Sherpa 3.0.0 (conda-forge). It uses the same Rivet analyses and exactly the scoring code of `hep/validate.py`.

**Setups.** The setups follow each program's own example inputs, changing only the beams, energy, generation cuts and output. Both run at leading order with parton showers; Sherpa merges extra jets (MEPS@LO). Their NLO modes need OpenLoops, which is not installed.

**Convention mismatches found and fixed.** Two of Round 7's findings are mismatches that would otherwise have made the comparison unfair:

- **Herwig's e⁺e⁻ setup includes initial-state radiation.** Photons came off the beams in 263 of 300 events, while ALEPH's data are corrected for it. The fix is `DoISR No` plus `NoPDF` on e±, verified to leave 0 beam photons in 300 events.
- **Sherpa's default leaves K⁰_S and Λ undecayed** (331 K⁰_S in 300 events). ALEPH counts their charged products, so the LEP runs decay particles with cτ < 300 mm (τ < 1 ns, the LEP convention).
- **My error, corrected:** Sherpa's minimum-bias card was first transcribed without its hadronization (`AHADIC`) block.

**Median χ²/ndf against data** (1 = agreement within uncertainties):

| Benchmark (measurements cited in Rounds 3–4) | SMLab LO (PYTHIA 8 Monash) | PYTHIA 8 direct | Herwig 7.3 | Sherpa 3.0 | SMLab best |
|---|---|---|---|---|---|
| ALEPH event shapes, 91.2 GeV | **2.48** | 2.47 | 19.48 † | 4.57 | 2.48 |
| ATLAS minimum bias, 13 TeV | 36.3 | 36.7 | **7.10** | not comparable ‡ | 36.3 |
| ATLAS Z pT, 13 TeV | 4.23 | 4.09 | 4.62 | 9.45 | **1.65** (FxFx + tune) |
| CMS t t̄ lepton + jets, 13 TeV | 7.92 | 8.67 | 15.59 | 15.44 | **1.82** (NLO + MadSpin + tune) |
| ATLAS t t̄ eμ, 13 TeV (Eur. Phys. J. C 80 (2020) 528) | 11.34 | — | 22.19 | 15.53 | **0.97** (NLO + MadSpin) |
| CMS inclusive jets, 13 TeV | **5.68** | 6.33 | 8.55 | 6.27 | 5.68 |

† **Herwig's LEP result is unconfirmed.** Its charged multiplicity is 19.48 against ALEPH's 20.91 ± 0.22 (ALEPH, Phys. Rept. 294 (1998) 1). A multiplicity below data is consistent with a trade-off the Herwig authors document (D. Reichelt, P. Richardson, A. Siodmok, Eur. Phys. J. C 77 (2017) 876, arXiv:1708.01491: LEP spectra and multiplicities cannot both be described well). But the poor event-shape agreement is surprising for a LEP-tuned generator. This build needed nine workarounds for Ubuntu 26.04's toolchain:
- Rust coreutils
- a libtool bug with conda's compilers
- GCC 15's stricter templates
- a missing `<cstdint>` include
- CMake 4
- the absence of zlib headers
- no system gfortran
- LHAPDF's Python wrapper
- the conda sysroot

A build artefact cannot be ruled out.

‡ **Sherpa's minimum bias is not comparable.** Sherpa's `Amisic` example is non-diffractive only (its diffractive and elastic parts are separate cards), while ATLAS's analysis requires diffraction to be included. The result is 62 % too many particles per event, so it is not a measure of Sherpa.

**What the comparison shows:**

- **Like for like at leading order,** SMLab's PYTHIA 8 matches or beats Herwig 7.3 and Sherpa 3.0 on every benchmark except minimum bias.
- **With its NLO samples** (MC@NLO, MadSpin, FxFx and the SMLab tune), SMLab is clearly best on Z pT and t t̄.
- **Minimum bias is SMLab's real weakness.** Herwig's model fits the 13 TeV charged-particle data far better (7.1 against 36). PYTHIA's Monash tune overshoots the multiplicity by 9 %, and SMLab offers no alternative there.
