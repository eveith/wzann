#include <vector>
#include <iostream>

#include <gtest/gtest.h>

#include <boost/range.hpp>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "SimpleWeightRandomizer.h"

#include "ElmanNetworkPattern.h"
#include "PerceptronNetworkPattern.h"

#include "TrainingSet.h"

#include "REvolutionaryTrainingAlgorithm.h"
#include "REvolutionaryTrainingAlgorithmTest.h"


using wzann::Layer;
using wzann::Neuron;
using wzann::Vector;
using wzann::Connection;
using wzann::TrainingSet;
using wzann::TrainingItem;
using wzann::NeuralNetwork;
using wzann::ActivationFunction;
using wzann::ElmanNetworkPattern;
using wzann::SimpleWeightRandomizer;
using wzann::PerceptronNetworkPattern;
using wzann::REvolutionaryTrainingAlgorithm;

using wzalgorithm::REvol;
using Individual = wzalgorithm::REvol::Individual;


NeuralNetwork* REvolutionaryTrainingAlgorithmTest::createNeuralNetwork()
{
    NeuralNetwork *net = new NeuralNetwork();

    ElmanNetworkPattern pattern;
    pattern.addLayer({ 2, ActivationFunction::Logistic });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    pattern.addLayer({ 1, ActivationFunction::Logistic });

    net->configure(pattern);
    return net;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualInitialization)
{
    auto* ann = createNeuralNetwork();
    Individual i1;
    REvolutionaryTrainingAlgorithm::getWeights(*ann, i1.parameters);

    int nConnections = 0;
    for (auto const& c : boost::make_iterator_range(ann->connections())) {
        if (!c->fixedWeight()) {
            nConnections++;
        }
    }

    ASSERT_EQ(nConnections, i1.parameters.size());

    delete ann;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testAgeIndividual)
{
    auto* ann = createNeuralNetwork();
    Individual i1;
    REvolutionaryTrainingAlgorithm::getWeights(*ann, i1.parameters);

    i1.timeToLive = 1;

    ASSERT_EQ(1l, i1.timeToLive);
    i1.age();
    ASSERT_EQ(0l, i1.timeToLive);

    delete ann;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualOperatorEquals)
{
    auto* ann = createNeuralNetwork();

    for (auto const& c : boost::make_iterator_range(ann->connections())) {
        if (! c->fixedWeight()) {
            c->weight(0.0);
        }
    }

    Individual i1, i2;
    REvolutionaryTrainingAlgorithm::getWeights(*ann, i1.parameters);
    REvolutionaryTrainingAlgorithm::getWeights(*ann, i2.parameters);

    ASSERT_TRUE(i1 == i2);

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](double& w) {
        w = 1.0;
    });

    ASSERT_FALSE((i1 == i2));

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](double& w) {
        w = 0.0;
    });

    i1.scatter.resize(i1.parameters.size());
    i2.scatter.resize(i2.parameters.size());

    i1.scatter[1] = 1.0;
    i2.scatter[1] = 1.1;

    ASSERT_FALSE((i1 == i2));
    i2.scatter[1] = 1.0;
    ASSERT_TRUE(i1 == i2);

    i1.restrictions.push_back(11.1);
    ASSERT_FALSE((i1 == i2));
    i2.restrictions.push_back(11.1);
    ASSERT_TRUE(i1 == i2);

    delete ann;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualOperatorAssign)
{
    auto* n1 = createNeuralNetwork(),
            *n2 = createNeuralNetwork();
    Individual i1, i2;
    REvolutionaryTrainingAlgorithm::getWeights(*n1, i1.parameters);
    REvolutionaryTrainingAlgorithm::getWeights(*n2, i2.parameters);

    if (i1 == i2) {
        i1.restrictions.push_back(421.43);
        i2.restrictions.push_back(-21.43);
    }

    ASSERT_FALSE((i1 == i2));

    i1 = i2;

    ASSERT_TRUE(i1 == i2);
    ASSERT_TRUE(&i1 != &i2);

    delete n1;
    delete n2;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testParametersSettingAndRetrieval)
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual;
    REvolutionaryTrainingAlgorithm::getWeights(
            *neuralNetwork,
            individual.parameters);

    std::vector<Connection*> connections;
    Vector parameters = individual.parameters;

    for (NeuralNetwork::size_type i = 0; i != neuralNetwork->size(); ++i) {
        Layer& layer = (*neuralNetwork)[i];

        for (Layer::size_type j = 0; j != layer.size(); ++j) {
            Neuron& neuron = layer[j];

            for (const auto &c : boost::make_iterator_range(
                     neuralNetwork->connectionsFrom(neuron))) {
                if (!c->fixedWeight()) {
                    connections.push_back(c);
                }
            }
        }
    }

    for (auto const& c : boost::make_iterator_range(
             neuralNetwork->connectionsFrom(neuralNetwork->biasNeuron()))) {
        if (! c->fixedWeight()) {
            connections.push_back(c);
        }
    }

    ASSERT_EQ(connections.size(), parameters.size());

    parameters.clear();
    for (size_t i = 0; i != connections.size(); ++i) {
        parameters.push_back(10.10);
    }

    individual.parameters = parameters;
    REvolutionaryTrainingAlgorithm::applyParameters(
            individual.parameters,
            *neuralNetwork);

    for (size_t i = 0; i != neuralNetwork->size(); ++i) {
        Layer *l = neuralNetwork->layerAt(i);

        for (size_t j = 0; j != l->size(); ++j) {
            Neuron *n = l->neuronAt(j);

            for (const auto &c: boost::make_iterator_range(
                     neuralNetwork->connectionsFrom(*n))) {
                if (! c->fixedWeight()) {
                    ASSERT_EQ(10.10, c->weight());
                }
            }
        }
    }

    delete neuralNetwork;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testTrainXOR)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern;
    pattern.addLayer({ 2, ActivationFunction::Identity });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    pattern.addLayer({ 1, ActivationFunction::Logistic });
    network.configure(pattern);
    SimpleWeightRandomizer().randomize(network);

    // Build training data:

    double targetVariance = 1e-2;
    double targetTrainingError = targetVariance * targetVariance / 9.;

    TrainingSet trainingSet;
    trainingSet.targetError(targetTrainingError).maxEpochs(100000)
            << TrainingItem({ 0.0, 0.0 }, { 0.0 })
            << TrainingItem({ 0.0, 1.0 }, { 1.0 })
            << TrainingItem({ 1.0, 0.0 }, { 1.0 })
            << TrainingItem({ 1.0, 1.0 }, { 0.0 });

    REvolutionaryTrainingAlgorithm trainingAlgorithm;
    trainingAlgorithm
            .populationSize(30)
            .eliteSize(3)
            .maxNoSuccessEpochs(trainingSet.maxEpochs())
            .startTTL(100)
            .gradientWeight(3.0)
            .successWeight(0.1)
            .ebmin(1e-2)
            .ebmax(2.0);
    trainingAlgorithm.train(network, trainingSet);

    std::cout << "Error: " << trainingSet.error()
            << ", Epochs: " << trainingSet.epochs() << "\n";

    Vector output;
    output = network.calculate({ 1., 1. });
    ASSERT_NEAR(0, output[0], targetVariance);
    output = network.calculate({ 1, 0 });
    ASSERT_NEAR(1, output[0], targetVariance);
    output = network.calculate({ 0, 0 });
    ASSERT_NEAR(0, output[0], targetVariance);
    output = network.calculate({ 0, 1 });
    ASSERT_NEAR(1, output[0], targetVariance);

    ASSERT_LE(trainingSet.error(), targetTrainingError);
    ASSERT_TRUE(trainingSet.epochs() < trainingSet.maxEpochs());
}
