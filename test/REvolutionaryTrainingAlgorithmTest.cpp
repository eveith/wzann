#include <QFile>
#include <QTextStream>

#include <iostream>

#include <gtest/gtest.h>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "SimpleWeightRandomizer.h"

#include "ElmanNetworkPattern.h"
#include "PerceptronNetworkPattern.h"
#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"

#include "TrainingSet.h"

#include "REvolutionaryTrainingAlgorithm.h"
#include "REvolutionaryTrainingAlgorithmTest.h"


using wzann::Layer;
using wzann::Neuron;
using wzann::Connection;
using wzann::Vector;
using wzann::NeuralNetwork;
using wzann::ElmanNetworkPattern;
using wzann::SimpleWeightRandomizer;
using wzann::PerceptronNetworkPattern;
using wzann::LinearActivationFunction;
using wzann::SigmoidActivationFunction;

using wzann::TrainingSet;
using wzann::TrainingItem;

using wzann::Individual;
using wzann::REvolutionaryTrainingAlgorithm;


NeuralNetwork* REvolutionaryTrainingAlgorithmTest::createNeuralNetwork()
{
    NeuralNetwork *net = new NeuralNetwork();

    static ElmanNetworkPattern pattern({
            2,
            3,
            1
        },
        {
            new SigmoidActivationFunction(),
            new SigmoidActivationFunction(),
            new SigmoidActivationFunction()
        });
    net->configure(pattern);

    return net;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualInitialization)
{
    NeuralNetwork *network = createNeuralNetwork();
    Individual i1(*network);

    int nConnections = 0;
    network->eachConnection([&nConnections](Connection *const &c) {
        if (!c->fixedWeight()) {
            nConnections++;
        }
    });

    ASSERT_EQ(nConnections, i1.parameters.size());
    ASSERT_EQ(nConnections, i1.scatter.size());

    delete network;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testAgeIndividual)
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(*neuralNetwork);

    individual.timeToLive = 1;

    ASSERT_EQ(1l, individual.timeToLive);
    individual.age();
    ASSERT_EQ(0l, individual.timeToLive);

    delete neuralNetwork;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualOperatorEquals)
{
    NeuralNetwork *network = createNeuralNetwork();

    network->eachConnection([](Connection *const &c) {
        if (! c->fixedWeight()) {
            c->weight(0.0);
        }
    });

    Individual i1(*network), i2(*network);

    ASSERT_TRUE(i1 == i2);

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](qreal &w) {
        w = 1.0;
    });

    ASSERT_FALSE((i1 == i2));

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](qreal &w) {
        w = 0.0;
    });

    i1.scatter[1] = 1.0;
    i2.scatter[1] = 1.1;

    ASSERT_FALSE((i1 == i2));
    i2.scatter[1] = 1.0;
    ASSERT_TRUE(i1 == i2);

    i1.errorVector()[0] = 11.1;
    ASSERT_FALSE((i1 == i2));
    i2.errorVector()[0] = 11.1;
    ASSERT_TRUE(i1 == i2);

    delete network;
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testIndividualOperatorAssign)
{
    NeuralNetwork *n1 = createNeuralNetwork(),
            *n2 = createNeuralNetwork();
    Individual i1(*n1), i2(*n2);

    if (i1 == i2) {
        i1.errorVector()[0] = 421.43;
        i2.errorVector()[0] = -21.43;
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
    Individual individual(*neuralNetwork);

    Vector parameters = individual.parameters;
    QList<Connection *> connections;

    for (NeuralNetwork::size_type i = 0; i != neuralNetwork->size(); ++i) {
        Layer &layer = (*neuralNetwork)[i];

        for (Layer::size_type j = 0; j != layer.size(); ++j) {
            Neuron &neuron = layer[j];

            for (const auto &c: boost::make_iterator_range(
                     neuralNetwork->connectionsFrom(neuron))) {
                if (!c->fixedWeight()) {
                    connections.push_back(c);
                }
            }
        }
    }

    for (Connection *c: boost::make_iterator_range(
             neuralNetwork->connectionsFrom(neuralNetwork->biasNeuron()))) {
        if (! c->fixedWeight()) {
            connections << c;
        }
    }

    ASSERT_EQ(connections.size(, parameters.size()));

    parameters.clear();
    for (int i = 0; i != connections.size(); ++i) {
        parameters << 10.10;
    }

    individual.parameters = parameters;
    Individual::applyParameters(individual, *neuralNetwork);

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


TEST_F(REvolutionaryTrainingAlgorithmTest, testModifyIndividual)
{
    REvolutionaryTrainingAlgorithm trainingAlgorithm;
    trainingAlgorithm.eliteSize(2).populationSize(5);

    REvolutionaryTrainingAlgorithm::Population population;
    for (size_t i = 0; i != trainingAlgorithm.populationSize(); ++i) {
        NeuralNetwork *n = createNeuralNetwork();
        population.push_back(new Individual(*n));
        delete n;
    }

    NeuralNetwork *n = createNeuralNetwork();
    population.push_back(new Individual(*n));
    auto &i3 = population.back();
    delete n;

    trainingAlgorithm.modifyWorstIndividual(
            population,
            /* currentSuccess = */ 0.2);

    ASSERT_EQ(
            population.front().parameters.size(),
            i3.parameters.size());

    std::for_each(population.begin(), population.end() - 1,
            [&](Winzent::Algorithm::detail::Individual &i) {
        for (auto j = 0; j != i.parameters.size(); ++j) {
            ASSERT_TRUE(i3.parameters.at(j) != i.parameters.at(j));
        }
    });
}


TEST_F(REvolutionaryTrainingAlgorithmTest, testTrainXOR)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern(
            {
                2,
                3,
                1
            }, {
                new LinearActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });
    network.configure(pattern);
    SimpleWeightRandomizer().randomize(network);

    // Build training data:

    QList<TrainingItem> trainingItems;
    trainingItems
            << TrainingItem({ 0.0, 0.0 }, { 0.0 })
            << TrainingItem({ 0.0, 1.0 }, { 1.0 })
            << TrainingItem({ 1.0, 0.0 }, { 1.0 })
            << TrainingItem({ 1.0, 1.0 }, { 0.0 });

    TrainingSet trainingSet(
            trainingItems,
            1e-2,
            15000);

    REvolutionaryTrainingAlgorithm trainingAlgorithm;
    trainingAlgorithm
            .populationSize(50)
            .eliteSize(5)
            .maxNoSuccessEpochs(trainingSet.maxEpochs())
            .startTTL(100)
            .gradientWeight(3.0)
            .successWeight(0.1)
            .ebmin(1e-2)
            .ebmax(2.0);

    QDateTime dt1 = QDateTime::currentDateTime();
    trainingAlgorithm.train(network, trainingSet);
    QDateTime dt2 = QDateTime::currentDateTime();

    qDebug() << "Trained XOR(x, y) in" << dt1.msecsTo(dt2) << "msec";

    auto output = network.calculate({ 1, 1 });
    qDebug() << "(1, 1) =>" << output;
    ASSERT_EQ(0, qRound(output[0]));
    output = network.calculate({ 1, 0 });
    qDebug() << "(1, 0) =>" << output;
    ASSERT_EQ(1, qRound(output[0]));
    output = network.calculate({ 0, 0 });
    qDebug() << "(0, 0) =>" << output;
    ASSERT_EQ(0, qRound(output[0]));
    output = network.calculate({ 0, 1 });
    qDebug() << "(0, 1) =>" << output;
    ASSERT_EQ(1, qRound(output[0]));

    QFile annDumpFile("testTrainXOR.out");
    annDumpFile.open(
            QIODevice::WriteOnly|QIODevice::Truncate|QIODevice::Text);
    annDumpFile.write(network.toJSON().toJson());
    annDumpFile.flush();
    annDumpFile.close();
}
