#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>

#include <boost/range.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>

#include "WzannGlobal.h"
#include "ClassRegistry.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"
#include "SimpleWeightRandomizer.h"


using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::unique_ptr;

using namespace wzann;
namespace po = boost::program_options;


void listAnnPatterns()
{
    cout << "Known ANN patterns:\n";

    auto* cr = ClassRegistry<NeuralNetworkPattern>::instance();
    for (auto const& rv : boost::make_iterator_range(cr->registry())) {
        cout << "  * " << rv.first << "\n";
    }
}


void listActivationFunctions()
{
    cout << "Known Activation Functions:\n";
    for (auto const* name : ActivationFunction::_names()) {
        cout << "  * " << name << "\n";
    }
}


unique_ptr<NeuralNetworkPattern> createAnnPattern(string const& name)
{
    auto* cr = ClassRegistry<NeuralNetworkPattern>::instance();

    if (! cr->isRegistered(name)) {
        return nullptr;
    }

    return unique_ptr<NeuralNetworkPattern>(cr->create(name));
}


NeuralNetworkPattern::SimpleLayerDefinition createSimpleLayerDefinition(
        string const& str)
{
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    NeuralNetworkPattern::SimpleLayerDefinition layerDefinition{
        0,
        ActivationFunction::Null };
    boost::char_separator<char> sep(":");
    tokenizer tokens(str, sep);

    if (std::distance(tokens.begin(), tokens.end()) != 2) {
        return layerDefinition;
    }

    auto tokIt = tokens.begin();
    layerDefinition.first = std::stoul(*tokIt++);
    layerDefinition.second = ActivationFunction::_from_string(tokIt->c_str());
    assert(++tokIt == tokens.end());

    return layerDefinition;
}


int main(int argc, char* argv[])
{
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("pattern,p",
                po::value<string>()->required(),
                "Sets the ANN pattern")
        ("add-layer,l",
                po::value<vector<string>>()->required(),
                "Adds a (simple) layer definition. "
                    "Format: NumNeurons:ActivationFunction")
        ("list-patterns,P", "Lists all available ANN patterns")
        ("list-activation-functions,A", "Lists all available "
            "activation functions")
        ("help,h", "Produces this help message")
        ("version,v", "Prints \"wzann-mkann " WZANN_VERSION "\"");
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc;
        return EXIT_SUCCESS;
    }

    if (vm.count("version")) {
        std::cout << "wzann-mkann " << WZANN_VERSION << "\n";
        return EXIT_SUCCESS;
    }

    if (vm.count("list-patterns") || vm.count("list-activation-functions")) {
        if (vm.count("list-patterns")) {
            listAnnPatterns();
        }

        if (vm.count("list-activation-functions")) {
            listActivationFunctions();
        }

        return EXIT_SUCCESS;
    }

    try {
        po::notify(vm); // Will raise on errors.
    } catch (po::required_option& e) {
        cerr << "ERROR: " << e.what() << ".\n";

        if (! vm.count("pattern")) {
            cerr
                    << "Please use '-p' to specify an ANN pattern.\n\n";
            listAnnPatterns();
            cerr << "\n\n";
        }

        cerr
                << "Run \"" << argv[0]
                << " --help\" to see all available options.\n";
        return EXIT_FAILURE;
    }

    auto pattern = createAnnPattern(vm.at("pattern").as<string>());
    if (! pattern) {
        cerr
                << "ANN pattern '"
                << vm.at("pattern").as<string>()
                << "' is unknown. Please specify a known pattern.\n\n";
        listAnnPatterns();
        return EXIT_FAILURE;
    }

    for (auto const& s : vm.at("add-layer").as<vector<string>>()) {
        auto layerDefinition = createSimpleLayerDefinition(s);

        if (0 == layerDefinition.first) {
            cerr << "Invalid number of neurons in layer definition.\n";
            return EXIT_FAILURE;
        }

        if (+ActivationFunction::Null == layerDefinition.second) {
            cerr
                    << "Invalid activation function specified "
                    << "in layer definition.\n\n";
            listActivationFunctions();
            return EXIT_FAILURE;
        }

        pattern->addLayer(layerDefinition);
    }

    auto ann = unique_ptr<NeuralNetwork>(new NeuralNetwork());
    ann->configure(*pattern);
    SimpleWeightRandomizer().randomize(*ann);
    cout << to_json(*ann);

    return EXIT_SUCCESS;
}
