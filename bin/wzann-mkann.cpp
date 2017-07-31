#include <iostream>

#include <boost/range.hpp>
#include <boost/program_options.hpp>

#include "WzannGlobal.h"
#include "ClassRegistry.h"
#include "NeuralNetworkPattern.h"


namespace po = boost::program_options;


void listAnnPatterns()
{
    std::cout << "Known ANN patterns:\n";

    auto* cr = wzann::ClassRegistry<wzann::NeuralNetworkPattern>::instance();
    for (auto const& rv : boost::make_iterator_range(cr->registry())) {
        std::cout << "  * " << rv.first << "\n";
    }
}


int main(int argc, char* argv[])
{
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("pattern,p", "Sets the ANN pattern")
        ("list-patterns,P", "Lists all available ANN patterns")
        ("help,h", "Produces this help message")
        ("version,v", "Prints \"wzann-mkann " WZANN_VERSION "\"");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc;
        return 0;
    }

    if (vm.count("version")) {
        std::cout << "wzann-mkann " << WZANN_VERSION << "\n";
        return 0;
    }

    if (vm.count("list-patterns")) {
        listAnnPatterns();
        return 0;
    }

    return 0;
}
