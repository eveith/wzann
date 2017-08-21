#include <memory>
#include <string>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <unistd.h>
#include <readline/history.h>
#include <readline/readline.h>

#include <boost/optional.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "Vector.h"
#include "WzannGlobal.h"
#include "NeuralNetwork.h"
#include "LayerSizeMismatchException.h"


using std::string;
using std::unique_ptr;

using namespace wzann;

namespace fs = boost::filesystem;
namespace po = boost::program_options;


unique_ptr<NeuralNetwork> readNeuralNetwork(string const& path)
{
    fs::path tsp(path);
    unique_ptr<NeuralNetwork> neuralNetwork(nullptr);

    if (! fs::exists(tsp)) {
        throw std::runtime_error(
                string("Neural network path '")
                    .append(path)
                    .append("' does not exist."));
    }

    std::ifstream infs(path);
    auto jsonString = static_cast<std::stringstream const&>(
            std::stringstream() << infs.rdbuf()).str();
    neuralNetwork.reset(new_from_json<NeuralNetwork>(jsonString));

    return neuralNetwork;
}


void writeNeuralNetwork(NeuralNetwork const& ann, string const& path)
{
    if (path != "-") {
        std::ofstream(path) << to_json(ann);
    } else {
        std::cout << to_json(ann);
    }
}


boost::optional<std::string> getlineReadline()
{
    boost::optional<std::string> ret;
    char* r = readline("> ");

    if (r) {
        ret.emplace(r);
        std::free(r);
    }

    return ret;
}


boost::optional<std::string> getlineUnbuffered()
{
    std::string s;
    std::getline(std::cin, s);

    boost::optional<std::string> ret;
    if (std::cin) {
        ret = s;
    }

    return ret;
}


Vector feedNeuralNetwork(NeuralNetwork& ann, string& input)
{
    std::remove_if(input.begin(), input.end(), [](char c) {
        switch (c) {
        case '(':
        case ')':
            return true;
        default:
            return false;
        }
    });

    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep(",;\t ");
    tokenizer tokens(input, sep);
    Vector result;

    if (std::distance(tokens.begin(), tokens.end())
            != ann.inputLayer().size()) {
        throw LayerSizeMismatchException(
                ann.inputLayer().size(),
                std::distance(tokens.begin(), tokens.end()));
    } else {
        Vector in;
        std::transform(
                tokens.begin(),
                tokens.end(),
                std::back_inserter(in),
                [](string const& s) -> double {
            return std::stod(s);
        });
        result = ann.calculate(in);
    }

    return result;
}


int main(int argc, char* argv[])
{
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("ann-input,i",
                po::value<string>()->required(),
                "Path to the ANN to read for REPL")
        ("ann-output,o",
                po::value<string>(),
                "Writes the ann with the new state, if desired.")
        ("help,h", "Produces this help message")
        ("version,v", "Prints \"wzann-repl " WZANN_VERSION "\"");

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc;
            return EXIT_SUCCESS;
        }

        if (vm.count("version")) {
            std::cout << "wzann-repl " << WZANN_VERSION << "\n";
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    } catch (po::error const& e) {
        std::cerr
                << "ERROR: " << e.what() << ".\n"
                << "Run \"" << argv[0]
                << " --help\" to see all available options.\n";
        return EXIT_FAILURE;
    }


    auto ann = readNeuralNetwork(vm["ann-input"].as<string>());

    auto* readlineFunction = &getlineUnbuffered;
    if (isatty(STDIN_FILENO)) { // Use readline
        std::cout
                << "wzann-repl "
                << WZANN_VERSION
                << "\n"
                << "Hit ^D to exit.\n";
        readlineFunction = &getlineReadline;
    }


    while (true) {
        auto in = (*readlineFunction)();
        if (! in.is_initialized()) {
            break;
        }

        try {
            auto const result = feedNeuralNetwork(*ann, *in);
            std::cout << result << "\n";
        } catch (wzann::LayerSizeMismatchException const& e) {
            std::cerr
                    << "ERROR: "
                    << e.what()
                    << " (" << e.expected() << " expected, "
                    << e.actual() << " given).\n";
        }
    }


    if (vm.count("ann-output")) {
        writeNeuralNetwork(*ann, vm["ann-output"].as<string>());
    }

    return EXIT_SUCCESS;
}
