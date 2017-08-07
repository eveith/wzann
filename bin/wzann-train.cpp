#include <cstdlib>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>

#include <boost/range.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "WzannGlobal.h"
#include "TrainingSet.h"
#include "ClassRegistry.h"
#include "TrainingAlgorithm.h"


#define EXIT_TRAINING_FAILURE (128+1)
#define EXIT_VERIFICYTION_FAILURE (EXIT_TRAINING_FAILURE+1)


using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::unique_ptr;

using namespace wzann;

using boost::make_iterator_range;

namespace fs = boost::filesystem;
namespace po = boost::program_options;



void listTrainingAlgorithms()
{
    cout << "Available training algorithms:\n";
    auto* cr = ClassRegistry<TrainingAlgorithm>::instance();
    for (auto const& i : make_iterator_range(cr->registry())) {
        cout << "  * " << i.first << "\n";
    }
}


unique_ptr<TrainingAlgorithm> createTrainingAlgorithm(string const& name)
{
    unique_ptr<TrainingAlgorithm> trainingAlgorithm(nullptr);
    auto* cr = ClassRegistry<TrainingAlgorithm>::instance();

    if (cr->isRegistered(name)) {
        trainingAlgorithm.reset(cr->create(name));
    } else {
        throw std::runtime_error("Unknown training algorithm");
    }

    return trainingAlgorithm;
}


unique_ptr<TrainingSet> readTrainingSet(string const& path)
{
    fs::path tsp(path);
    unique_ptr<TrainingSet> trainingSet(nullptr);

    if (! fs::exists(tsp)) {
        throw std::runtime_error(
                string("Training set path '")
                    .append(path)
                    .append("' does not exist."));
    }

    std::ifstream infs(path);
    auto jsonString = static_cast<std::stringstream const&>(
            std::stringstream() << infs.rdbuf()).str();
    trainingSet.reset(new_from_json<TrainingSet>(jsonString));

    return trainingSet;
}


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


void writeAnn(NeuralNetwork const& ann, string const& path)
{
    if (path != "-") {
        std::ofstream(path) << to_json(ann);
    } else {
        cout << to_json(ann);
    }
}


double runVerificationSet(NeuralNetwork& ann, TrainingSet const& vs)
{
    double error = 0.0;
    size_t numRelevantItems = 0;

    for (auto const& vi : vs.trainingItems) {
        const auto actual = ann.calculate(vi.input());
        if (! vi.outputRelevant()) {
            continue;
        }
        numRelevantItems++;
        auto const& expected = vi.expectedOutput();

        double lerror = 0.0;
        for (auto ait = actual.begin(), eit = expected.begin();
                ait != actual.end() && eit != expected.end();
                ait++, eit++) {
            lerror += std::pow(*eit - *ait, 2);
        }

        error += lerror / 2.0;
    }

    return error / static_cast<double>(numRelevantItems);
}


int main (int argc, char* argv[])
{
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("ann-input,i",
                po::value<string>()->required(),
                "The input ANN to train.")
        ("training-set-input,I",
                po::value<string>()->required(),
                "Input training set")
        ("ann-output,o",
                po::value<string>()->default_value("-"),
                "Where to output the trained ANN to. "
                    "Defaults to STDOUT ('-').")
        ("verify-input,V",
                po::value<string>(),
                "The path of the training set used for verification.")
        ("target-error,e",
                po::value<double>(),
                "The desired training error; taken from the training set "
                    "if not specified")
        ("max-epochs,E",
                po::value<wzann::TrainingAlgorithm::epoch_t>(),
                "The maximum number of iterations to run the training; "
                    "taken from the training set if not specified")
        ("training-algorithm,t",
                po::value<string>()->required(),
                "Chooses the appropriate training algorithm")
        ("list-training-algorithms,T",
            "Lists all available training algorithms")
        ("help,h", "Produces this help message")
        ("version,v", "Prints \"wzann-train " WZANN_VERSION "\"");

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);


        if (vm.count("help")) {
            std::cout
                    << desc
                    << "\nReturns "
                    << EXIT_SUCCESS
                    << " on success, "
                    << EXIT_FAILURE
                    << " on errors caused by malformed input, "
                    << EXIT_TRAINING_FAILURE
                    <<  ", if the training was unsuccessful, and "
                    << EXIT_VERIFICYTION_FAILURE
                    << " if the training was successful, but the error "
                        "obtained by the verification data set exceeded the "
                        "desired target error.\n";
            return EXIT_SUCCESS;
        }

        if (vm.count("version")) {
            std::cout << "wzann-train " << WZANN_VERSION << "\n";
            return EXIT_SUCCESS;
        }

        if (vm.count("list-training-algorithms")) {
            listTrainingAlgorithms();
            return EXIT_SUCCESS;
        }


        po::notify(vm); // Will raise on errors.
    } catch (po::error const& e) {
        cerr
                << "ERROR: " << e.what() << ".\n"
                << "Run \"" << argv[0]
                << " --help\" to see all available options.\n";
        return EXIT_FAILURE;
    }


    unique_ptr<TrainingSet> trainingSet;
    unique_ptr<TrainingSet> verificationSet;
    unique_ptr<NeuralNetwork> neuralNetwork;
    unique_ptr<TrainingAlgorithm> trainingAlgorithm;

    try {
        trainingAlgorithm = createTrainingAlgorithm(vm.at(
                "training-algorithm").as<string>());
        trainingSet = readTrainingSet(
                vm.at("training-set-input").as<string>());
        neuralNetwork = readNeuralNetwork(
                vm.at("ann-input").as<string>());

        if (vm.count("verify-input")) {
            verificationSet = readTrainingSet(vm.at("").as<string>());
        }
    } catch (std::exception& e) {
        cerr << "ERROR: " << e.what() << ".\n";
        return EXIT_FAILURE;
    }


    trainingAlgorithm->train(*neuralNetwork, *trainingSet);
    cerr
            << "Training ended. Final error: "
            << trainingSet->error() << "/" << trainingSet->targetError()
            << ", number of epochs taken: "
            << trainingSet->epochs() << "/" << trainingSet->maxEpochs();


    writeAnn(*neuralNetwork, vm.at("ann-output").as<string>());


    if (verificationSet) {
        auto verror = runVerificationSet(*neuralNetwork, *verificationSet);
        cerr
                << "Verification set error: "
                << verificationSet->error()
                << "/"
                << verror
                << "\n";
        if (verror > verificationSet->targetError()) {
            return EXIT_VERIFICYTION_FAILURE;
        }
    } else if (trainingSet->error() > trainingSet->targetError()) {
        return EXIT_TRAINING_FAILURE;
    }
    return EXIT_SUCCESS;
}
