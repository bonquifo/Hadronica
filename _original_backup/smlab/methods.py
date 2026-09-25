"""What the laboratory actually computes, written for the in-app methods panel."""

METHODS = """
SMLab generates leading-order Standard Model collisions and draws them in a schematic solenoidal detector. It is a precision teaching instrument, not a substitute for MadGraph, PYTHIA, or Geant4.

CONSTANTS
Masses and widths follow the 2024 Review of Particle Physics (S. Navas et al., Phys. Rev. D 110, 030001). The fine-structure constant is the CODATA value α⁻¹(0) = 137.035999084. The weak mixing angle in the couplings is the effective leptonic value sin²θ_eff = 0.23153. G_F = 1.1663788×10⁻⁵ GeV⁻² and α_s(M_Z) = 0.1180.

FERMION PAIRS
e⁺e⁻ → f f̄, and the parton-level Drell–Yan process q q̄ → f f̄, use the improved Born approximation: photon exchange, Z exchange, and γZ interference. The reduced propagator is

χ(s) = [G_F M_Z² / (2√2 π α)] · s / (s − M_Z² + i s Γ_Z / M_Z)

with g_A = T₃ and g_V = T₃ − 2 Q sin²θ_eff. On the Z pole this resonant piece reproduces the Breit–Wigner built from the same partial widths. Vector couplings turn on as β(3−β²)/2 and axial couplings as β³. Quark final states are multiplied by N_c (1 + α_s(s)/π). θ is the angle between the incoming fermion and the outgoing fermion.

This is the effective-Born structure used in LEP comparisons. It is not a complete O(α) electroweak library: there are no weak form factors and no exclusive hadronic resonances.

INITIAL-STATE RADIATION
For lepton colliders a leading-log Kuraev–Fadin radiator folds the Born cross section. One collinear photon represents that radiation. It escapes down the beam pipe, the hard scale becomes s′ = s(1−v), and the hard system recoils. That is the radiative-return tail above the Z.

BHABHA AND TWO PHOTONS
e⁺e⁻ → e⁺e⁻ is the QED Born cross section with photon exchange only, cut to 10° < θ < 170° so the forward pole is finite. The Z diagrams of Bhabha are omitted, so do not read a Z-pole asymmetry from this channel. e⁺e⁻ → γγ uses the massless QED result with the identical-photon factor 1/2 and the same angular cut.

HIGGSSTRAHLUNG
e⁺e⁻ → ZH uses the Born formula

σ = [G_F² M_Z⁴ / (96π s)] (v_e² + a_e²) β (β² + 12 M_Z²/s) / (1 − M_Z²/s)²

with a_e = −1 and v_e = −1 + 4 sin²θ_eff (arXiv:hep-ph/9512355). A one-loop α(0)-scheme benchmark is 225.6 fb at 250 GeV (Phys. Rev. D 100, 073002). This leading-order G_F-scheme result is a few percent higher. The Z angular weight is β² sin²θ + 8 M_Z²/s, normalized to that total rate.

DECAYS
Z and W branching fractions are computed from these couplings. W quark channels use PDG CKM magnitudes and a factor 1 + α_s/π. Higgs branching fractions are the Standard Model table near 125 GeV (LHC Higgs Cross Section Working Group, as quoted by the PDG review), renormalized to one. H → WW* and H → ZZ* sample virtual masses from a q² Breit–Wigner times phase space; 125 GeV is below both pair thresholds.

Tau decays use PDG exclusive fractions for the leptonic, pion, rho, and a1 channels. The remaining few percent, mostly kaons, is represented by a charge-conserving four-pion state. Leptonic muon and tau decays use the Michel spectrum. W, Z, and Higgs decay angles are isotropic in the parent rest frame: full V−A spin correlations are not simulated. Muons are stable unless “decay muons” is on, which matches a real tracker (cτ_μ = 659 m).

TIMES
After the two particles meet, the report lists every product. A mean life is cτ/c from the PDG, or ħ/Γ from the PDG width. The Higgs entry uses the Standard Model width 4.07 MeV quoted by the PDG, not the loosely measured width. Quarks other than the top, and gluons, are partons and have no free-particle lifetime in this program. The approach on screen is slowed so the paths can be watched. The hard-scatter time is ħ/√s′, not the length of that picture.

DETECTOR
Charged tracks obey p_T = 0.299792458 |q| B R in a uniform 3.8 T field along the beam, the CMS solenoid value. Positive charges curve clockwise in the transverse view. The layering — tracker, ECAL, HCAL, muon stations — is a schematic barrel, not a Geant4 geometry. There is no material interaction, no resolution smearing, and no nuclear shower. Quarks and gluons are drawn as partons. Neutrinos appear only as missing transverse momentum.

The 3D button draws that same barrel. A charged track is a helix: the transverse circle is the formula above, and the distance along the field is (p_z / p_T) times the arc length. Straight tracks run until they meet a cylinder wall or the end of that layer. Inner half-lengths are R sinh(1.5). The gold coil is 12.5 m long, the CMS solenoid, at the radius already used in the slice. Dragging the picture only moves the camera.

CUSTOM INCOMING STATE
Custom chooses the two incoming particles, a momentum for each, and the angle between those momenta. 180° is head-on. Speed is β = |p|/E and is displayed; it is not a separate dial, because mass and momentum already fix it. √s is the invariant mass of the two 4-vectors. Products are generated with the same Born formulas as the preset beams, in the center of mass, then rotated and boosted into the lab frame you defined. Pairs without a formula — anything other than e⁻e⁺, μ⁻μ⁺, or a quark with its antiquark — are drawn as incoming rays only.

NOT INCLUDED
Parton showers, hadronization, PDFs, beam polarization, beamstrahlung, full one-loop electroweak corrections, and e⁺e⁻ → W⁺W⁻. Proton collisions are not offered, because a proton is not an elementary initial state in this Born treatment.
""".strip()
