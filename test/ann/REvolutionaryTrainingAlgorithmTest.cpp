#include <QtTest>
#include <QObject>

#include <QFile>
#include <QTextStream>

#include <iostream>

#include "Testrunner.h"

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


using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;
using Winzent::ANN::Connection;
using Winzent::ANN::Vector;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::ElmanNetworkPattern;
using Winzent::ANN::SimpleWeightRandomizer;
using Winzent::ANN::PerceptronNetworkPattern;
using Winzent::ANN::LinearActivationFunction;
using Winzent::ANN::SigmoidActivationFunction;

using Winzent::ANN::TrainingSet;
using Winzent::ANN::TrainingItem;

using Winzent::ANN::Individual;
using Winzent::ANN::REvolutionaryTrainingAlgorithm;


REvolutionaryTrainingAlgorithmTest::REvolutionaryTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


NeuralNetwork *REvolutionaryTrainingAlgorithmTest::createNeuralNetwork()
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


void REvolutionaryTrainingAlgorithmTest::testIndividualInitialization()
{
    NeuralNetwork *network = createNeuralNetwork();
    Individual i1(*network);

    int nConnections = 0;
    network->eachConnection([&nConnections](Connection *const &c) {
        if (!c->fixedWeight()) {
            nConnections++;
        }
    });

    QCOMPARE(i1.parameters.size(), nConnections);
    QCOMPARE(i1.scatter.size(), nConnections);

    delete network;
}


void REvolutionaryTrainingAlgorithmTest::testAgeIndividual()
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(*neuralNetwork);

    individual.timeToLive = 1;

    QCOMPARE(individual.timeToLive, 1l);
    individual.age();
    QCOMPARE(individual.timeToLive, 0l);

    delete neuralNetwork;
}


void REvolutionaryTrainingAlgorithmTest::testIndividualOperatorEquals()
{
    NeuralNetwork *network = createNeuralNetwork();

    network->eachConnection([](Connection *const &c) {
        if (! c->fixedWeight()) {
            c->weight(0.0);
        }
    });

    Individual i1(*network), i2(*network);

    QVERIFY(i1 == i2);

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](qreal &w) {
        w = 1.0;
    });

    QVERIFY(! (i1 == i2));

    std::for_each(i2.parameters.begin(), i2.parameters.end(), [](qreal &w) {
        w = 0.0;
    });

    i1.scatter[1] = 1.0;
    i2.scatter[1] = 1.1;

    QVERIFY(! (i1 == i2));
    i2.scatter[1] = 1.0;
    QVERIFY(i1 == i2);

    i1.errorVector()[0] = 11.1;
    QVERIFY(! (i1 == i2));
    i2.errorVector()[0] = 11.1;
    QVERIFY(i1 == i2);

    delete network;
}


void REvolutionaryTrainingAlgorithmTest::testIndividualOperatorAssign()
{
    NeuralNetwork *n1 = createNeuralNetwork(),
            *n2 = createNeuralNetwork();
    Individual i1(*n1), i2(*n2);

    if (i1 == i2) {
        i1.errorVector()[0] = 421.43;
        i2.errorVector()[0] = -21.43;
    }

    QVERIFY(! (i1 == i2));

    i1 = i2;

    QVERIFY(i1 == i2);
    QVERIFY(&i1 != &i2);

    delete n1;
    delete n2;
}


void REvolutionaryTrainingAlgorithmTest::testParametersSettingAndRetrieval()
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(*neuralNetwork);

    Vector parameters = individual.parameters;
    QList<Connection *> connections;

    for (size_t i = 0; i != neuralNetwork->size(); ++i) {
        Layer *l = neuralNetwork->layerAt(i);

        for (size_t j = 0; j != l->size(); ++j) {
            Neuron *n = l->neuronAt(j);

            foreach (Connection *c, neuralNetwork->neuronConnectionsFrom(n)) {
                if (!c->fixedWeight()) {
                    connections.push_back(c);
                }
            }
        }
    }

    QList<Connection *> biasConnections =
            neuralNetwork->neuronConnectionsFrom(neuralNetwork->biasNeuron());
    foreach (Connection *c, biasConnections) {
        if (! c->fixedWeight()) {
            connections << c;
        }
    }

    QCOMPARE(parameters.size(), connections.size());

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

            for (const auto &c: neuralNetwork->neuronConnectionsFrom(n)) {
                if (! c->fixedWeight()) {
                    QCOMPARE(c->weight(), 10.10);
                }
            }
        }
    }

    delete neuralNetwork;
}


void REvolutionaryTrainingAlgorithmTest::testModifyIndividual()
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
    Individual i3(*n);
    delete n;
    trainingAlgorithm.modifyIndividual(
                static_cast<Winzent::Algorithm::detail::Individual &>(i3),
                population);

    QCOMPARE(
            i3.parameters.size(),
            population.front().parameters.size());

    std::for_each(population.begin(), population.end() - 1,
            [&](Individual const& i) {
        for (auto j = 0; j != i.parameters.size(); ++j) {
            QVERIFY (i3.parameters.at(j) != i.parameters.at(j));
        }
    });
}


void REvolutionaryTrainingAlgorithmTest::testSortPopulation()
{
    NeuralNetwork *n1 = createNeuralNetwork(),
            *n2 = createNeuralNetwork(),
            *n3 = createNeuralNetwork();

    Individual *i1 = new Individual(*n1);
    Individual *i2 = new Individual(*n2);
    Individual *i3 = new Individual(*n3);

    i1->timeToLive = 10;
    i1->errorVector()[0] = 0.25;

    i2->timeToLive = 10;
    i2->errorVector()[0] = 0.5;

    i3->timeToLive = 2;
    i3->errorVector()[0] = 0.5;

    QVERIFY(i1->isBetterThan(*i2));

    Winzent::Algorithm::REvol::Population population;
    population.push_back(i2);
    population.push_back(i1);
    population.push_back(i3);

    QCOMPARE(&(population.front()), i2);
    Winzent::Algorithm::REvol::sortPopulation(population);
    QCOMPARE(&(population.at(0)), i1);
    QCOMPARE(&(population.at(1)), i2);
    QCOMPARE(&(population.at(2)), i3);

    delete n1;
    delete n2;
    delete n3;
}


void REvolutionaryTrainingAlgorithmTest::testTrainXOR()
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
            5000);

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
    QCOMPARE(qRound(output[0]), 0);
    output = network.calculate({ 1, 0 });
    qDebug() << "(1, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 1);
    output = network.calculate({ 0, 0 });
    qDebug() << "(0, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 0);
    output = network.calculate({ 0, 1 });
    qDebug() << "(0, 1) =>" << output;
    QCOMPARE(qRound(output[0]), 1);

    QFile annDumpFile("testTrainXOR.out");
    annDumpFile.open(
            QIODevice::WriteOnly|QIODevice::Truncate|QIODevice::Text);
    annDumpFile.write(network.toJSON().toJson());
    annDumpFile.flush();
    annDumpFile.close();
}


TESTCASE(REvolutionaryTrainingAlgorithmTest)
