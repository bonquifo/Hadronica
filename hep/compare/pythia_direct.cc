// PYTHIA 8 run directly into Rivet, bypassing every Hadronica layer (worker, event
// serialization, Hadronica's HepMC conversion, weight handling). Events are converted
// by PYTHIA's own HepMC3 interface (Pythia8Plugins/HepMC3.h) and analysed by a
// Rivet 4 AnalysisHandler. (PYTHIA 8.312's Pythia8Rivet.h targets the Rivet 3 API.)
//
//   pythia_direct <settings.cmnd> <analyses,comma,separated> <events> <seed> <output.yoda>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "HepMC3/GenEvent.h"
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Rivet/AnalysisHandler.hh"

int main(int argc, char* argv[]) {
    if (argc != 6) return 2;
    Pythia8::Pythia pythia;
    pythia.readFile(argv[1]);
    pythia.readString("Random:setSeed = on");
    pythia.readString(std::string("Random:seed = ") + argv[4]);
    if (!pythia.init()) return 1;

    std::vector<std::string> analyses;
    std::stringstream list(argv[2]);
    for (std::string name; std::getline(list, name, ',');) analyses.push_back(name);
    Rivet::AnalysisHandler rivet;
    rivet.addAnalyses(analyses);

    HepMC3::Pythia8ToHepMC3 converter;
    const long events = std::atol(argv[3]);
    for (long done = 0; done < events;) {
        if (!pythia.next()) continue;
        HepMC3::GenEvent event(HepMC3::Units::GEV, HepMC3::Units::MM);
        converter.fill_next_event(pythia, &event, done);
        rivet.analyze(event);
        ++done;
    }
    rivet.finalize();
    rivet.writeData(argv[5]);
    return 0;
}
