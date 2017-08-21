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
#include "REvolutionaryTrainingAlgorithm.h"


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



po::options_description buildCliOptions()
{
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
        ("revol-population-size",
                po::value<size_t>()->default_value(30),
                "REvol: Size of the general population")
        ("revol-elite-size",
                po::value<size_t>()->default_value(3),
                "REvol: The size of the elite (contained in the population)")
        ("revol-gradient-weight",
                po::value<double>()->default_value(1.0),
                "REvol: Weight of the implicit gradient information")
        ("revol-success-weight",
                po::value<double>()->default_value(1.0),
                "REvol: Weight of population success rate")
        ("revol-measurement-epochs",
                po::value<wzalgorithm::REvol::epoch_t>()->default_value(
                    REvolutionaryTrainingAlgorithm().measurementEpochs()),
                "REvol: Time period of the pt1 function used for measuring "
                    "overall population success")
        ("revol-max-no-success-epochs",
                po::value<wzalgorithm::REvol::epoch_t>()->default_value(
                    std::numeric_limits<wzalgorithm::REvol::epoch_t>::max()),
                "REvol: Maximum number of epochs without a global success")
        ("revol-startttl",
                po::value<std::ptrdiff_t>()->default_value(5 * 30),
                "REvol: Initial Time To Live of a new individual")
        ("revol-eamin",
                po::value<double>()->default_value(
                    REvolutionaryTrainingAlgorithm().eamin()),
                "REvol: Absolute minimum value of a change")
        ("revol-ebmin",
                po::value<double>()->default_value(
                    REvolutionaryTrainingAlgorithm().ebmin()),
                "REvol: Relative minimum value of a change")
        ("revol-ebmax",
                po::value<double>()->default_value(
                    REvolutionaryTrainingAlgorithm().ebmax()),
                "REvol: Relative maximum value of a change")
        ("help,h", "Produces this help message")
        ("version,v", "Prints \"wzann-train " WZANN_VERSION "\"");

    return desc;
}



void printHelp(
        po::options_description const& desc,
        std::ostream& os = std::cout)
{
    os
            << desc
            << "\n"
            << "Returns   "
            << EXIT_SUCCESS
            << " on success,\n          "
            << EXIT_FAILURE
            << " on errors caused by malformed input,\n        "
            << EXIT_TRAINING_FAILURE
            <<  " if the training was unsuccessful, and\n        "
            << EXIT_VERIFICYTION_FAILURE
            << " if the training was successful, but the error "
                "obtained\n            by the verification data set "
                "exceeded the desired target error.\n";
}



void listTrainingAlgorithms()
{
    cout << "Available training algorithms:\n";
    auto* cr = ClassRegistry<TrainingAlgorithm>::instance();
    for (auto const& i : make_iterator_range(cr->registry())) {
        cout << "  * " << i.first << "\n";
    }
}


template <class T>
void configureTrainingAlgorithm(
        T&,
        po::variables_map const& = po::variables_map())
{
}


template <>
void configureTrainingAlgorithm(
        REvolutionaryTrainingAlgorithm& trainingAlgorithm,
        boost::program_options::variables_map const& vm)
{
    trainingAlgorithm
            .successWeight(vm["revol-success-weight"].as<double>())
            .measurementEpochs(vm["revol-measurement-epochs"].as<
                wzalgorithm::REvol::epoch_t>())
            .gradientWeight(vm["revol-gradient-weight"].as<double>())
            .populationSize(vm["revol-population-size"].as<size_t>())
            .eliteSize(vm["revol-elite-size"].as<size_t>())
            .startTTL(vm["revol-startttl"].as<std::ptrdiff_t>())
            .eamin(vm["revol-eamin"].as<double>())
            .ebmin(vm["revol-ebmin"].as<double>())
            .ebmax(vm["revol-ebmax"].as<double>())
            .maxNoSuccessEpochs(vm["revol-max-no-success-epochs"].as<
                wzalgorithm::REvol::epoch_t>());
}


unique_ptr<TrainingAlgorithm> createTrainingAlgorithm(
        string const& name,
        po::variables_map const& commandLineArguments = po::variables_map())
{
    unique_ptr<TrainingAlgorithm> trainingAlgorithm(nullptr);
    auto* cr = ClassRegistry<TrainingAlgorithm>::instance();

    if (cr->isRegistered(name)) {
        trainingAlgorithm.reset(cr->create(name));

        if (name == "wzann::REvolutionaryTrainingAlgorithm") {
            configureTrainingAlgorithm<REvolutionaryTrainingAlgorithm>(
                    dynamic_cast<REvolutionaryTrainingAlgorithm&>(
                        *trainingAlgorithm),
                    commandLineArguments);
        }
    } else {
        throw std::runtime_error("Unknown training algorithm");
    }

    return trainingAlgorithm;
}


unique_ptr<TrainingSet> readTrainingSet(
        string const& path,
        po::variables_map const& options)
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

    if (options.count("target-error")) {
        trainingSet->targetError(options.at("target-error").as<double>());
    }
    if (options.count("max-epochs")) {
        trainingSet->maxEpochs(options.at("max-epochs").as<size_t>());
    }

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


void writeNeuralNetwork(NeuralNetwork const& ann, string const& path)
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
    auto desc(buildCliOptions());


    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            printHelp(desc);
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

        po::notify(vm);
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
                "training-algorithm").as<string>(), vm);
        trainingSet = readTrainingSet(
                vm.at("training-set-input").as<string>(),
                vm);
        neuralNetwork = readNeuralNetwork(
                vm.at("ann-input").as<string>());

        if (vm.count("verify-input")) {
            verificationSet = readTrainingSet(
                    vm.at("verify-input").as<string>(),
                    vm);
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
            << trainingSet->epochs() << "/" << trainingSet->maxEpochs()
            << "\n";


    writeNeuralNetwork(*neuralNetwork, vm.at("ann-output").as<string>());


    if (verificationSet) {
        auto verror = runVerificationSet(*neuralNetwork, *verificationSet);
        cerr
                << "Verification set error: "
                << verror
                << "/"
                << verificationSet->targetError()
                << "\n";
        if (verror > verificationSet->targetError()) {
            return EXIT_VERIFICYTION_FAILURE;
        }
    } else if (trainingSet->error() > trainingSet->targetError()) {
        return EXIT_TRAINING_FAILURE;
    }
    return EXIT_SUCCESS;
}
