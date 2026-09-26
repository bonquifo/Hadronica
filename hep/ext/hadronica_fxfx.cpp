// Attach PYTHIA's FxFx jet-matching veto to a Pythia object created in Python.
//
// PYTHIA's JetMatching:* settings (doFxFx, qCut, nJetMax, ...) only act through
// the JetMatchingMadgraph user hook in Pythia8Plugins/JetMatching.h, which the
// Python bindings do not expose. CombineMatchingInput sets that hook up exactly
// as PYTHIA's own FxFx example program (main164, formerly main89) does. The
// module shares pybind11 internals (v12) with pythia8.so, so it accepts the
// Pythia8::Pythia objects that module creates. Call attach() before init().
//
// FxFx: R. Frederix, S. Frixione, JHEP 12 (2012) 061, arXiv:1209.6215.

#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/CombineMatchingInput.h"

namespace {

// The hooks must outlive the Pythia objects that use them.
std::vector<std::unique_ptr<Pythia8::CombineMatchingInput>> combiners;

void attach(Pythia8::Pythia& pythia) {
    // setHook() asks whether the input is an Alpgen file; register the key as unset.
    if (!pythia.settings.isWord("Alpgen:file")) pythia.settings.addWord("Alpgen:file", "void");
    combiners.emplace_back(std::make_unique<Pythia8::CombineMatchingInput>());
    combiners.back()->setHook(pythia);
}

}  // namespace

PYBIND11_MODULE(hadronica_fxfx, m) {
    m.doc() = "PYTHIA 8 FxFx jet-matching hook for Hadronica";
    m.def("attach", &attach, "Attach the FxFx/MLM jet-matching user hook (before pythia.init()).");
}
