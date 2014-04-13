#include <QtTest>

#include <QObject>

#include "Testrunner.h"

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ElmanNetworkPattern.h"
#include "SigmoidActivationFunction.h"
#include "NguyenWidrowWeightRandomizer.h"
#include "REvolutionaryTrainingAlgorithm.h"
#include "TrainingSet.h"

#include "REvolutionaryTrainingAlgorithmTest.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;
using Winzent::ANN::Connection;
using Winzent::ANN::ElmanNetworkPattern;
using Winzent::ANN::SigmoidActivationFunction;
using Winzent::ANN::NguyenWidrowWeightRandomizer;
using Winzent::ANN::Individual;
using Winzent::ANN::REvolutionaryTrainingAlgorithm;
using Winzent::ANN::TrainingSet;


REvolutionaryTrainingAlgorithmTest::REvolutionaryTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


NeuralNetwork *REvolutionaryTrainingAlgorithmTest::createNeuralNetwork()
{
    NeuralNetwork *net = new NeuralNetwork(this);

    static ElmanNetworkPattern pattern({
            2,
            3,
            1
        },
        {
            new SigmoidActivationFunction(1.0, this),
            new SigmoidActivationFunction(1.0, this),
            new SigmoidActivationFunction(1.0, this)
        });
    net->configure(&pattern);

    static NguyenWidrowWeightRandomizer randomizer;
    randomizer.randomize(net);

    return net;
}


void REvolutionaryTrainingAlgorithmTest::testIndividualInitialization()
{
    NeuralNetwork *network = createNeuralNetwork();
    Individual i1(network);

    int nConnections = 0;
    network->eachConnection([&nConnections](Connection *const &c) {
        if (!c->fixedWeight()) {
            nConnections++;
        }
    });

    QCOMPARE(i1.parameters().size(), nConnections);
    QCOMPARE(i1.scatter().size(), nConnections);
}


void REvolutionaryTrainingAlgorithmTest::testAgeIndividual()
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(neuralNetwork);

    individual.timeToLive(1);

    QCOMPARE(individual.timeToLive(), 1);
    QVERIFY(individual.isAlive());
    individual.age();
    QCOMPARE(individual.timeToLive(), 0);
    QVERIFY(!individual.isAlive());
}


void REvolutionaryTrainingAlgorithmTest::testParametersSettingAndRetrieval()
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(neuralNetwork);

    QList<qreal> parameters = individual.parameters();
    QList<Connection *> connections;

    for (int i = 0; i != neuralNetwork->size(); ++i) {
        Layer *l = neuralNetwork->layerAt(i);

        for (int j = 0; j != l->size() + 1; ++j) {
            Neuron *n = l->neuronAt(j);

            foreach (Connection *c, neuralNetwork->neuronConnectionsFrom(n)) {
                if (!c->fixedWeight()) {
                    connections << c;
                }
            }
        }
    }

    QCOMPARE(parameters.size(), connections.size());

    parameters.clear();
    for (int i = 0; i != connections.size(); ++i) {
        parameters << 10.10;
    }

    individual.parameters(parameters);

    for (int i = 0; i != neuralNetwork->size(); ++i) {
        Layer *l = neuralNetwork->layerAt(i);

        for (int j = 0; j != l->size(); ++j) {
            Neuron *n = l->neuronAt(j);

            foreach (Connection *c, neuralNetwork->neuronConnectionsFrom(n)) {
                if (!c->fixedWeight()) {
                    QCOMPARE(c->weight(), 10.10);
                }
            }
        }
    }
}


void REvolutionaryTrainingAlgorithmTest::testGenerateIndividual()
{
    NeuralNetwork *n1 = createNeuralNetwork(), *n2 = createNeuralNetwork();
    Individual i1(n1), i2(n2);

    REvolutionaryTrainingAlgorithm trainingAlgorithm(n1);
    trainingAlgorithm.eliteSize(1).populationSize(2);
    TrainingSet trainingSet({ }, 1.0, 1000);

    Individual *i3 = trainingAlgorithm.generateIndividual(
            { &i1, &i2 }, &trainingSet);

    QList<qreal> parametersI1 = i1.parameters();
    QList<qreal> parametersI2 = i2.parameters();
    QList<qreal> parametersI3 = i3->parameters();

    QCOMPARE(parametersI3.size(), parametersI1.size());
    QCOMPARE(parametersI3.size(), parametersI2.size());

    for (int i = 0; i != parametersI3.size(); ++i) {
        QVERIFY(parametersI3.at(i) != parametersI1.at(i));
        QVERIFY(parametersI3.at(i) != parametersI2.at(i));
    }
}


TESTCASE(REvolutionaryTrainingAlgorithmTest)
