#include <QtTest>

#include <QObject>

#include "Testrunner.h"

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ElmanNetworkPattern.h"
#include "PerceptronNetworkPattern.h"
#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"
#include "REvolutionaryTrainingAlgorithm.h"
#include "TrainingSet.h"

#include "REvolutionaryTrainingAlgorithmTest.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;
using Winzent::ANN::Connection;
using Winzent::ANN::ElmanNetworkPattern;
using Winzent::ANN::PerceptronNetworkPattern;
using Winzent::ANN::LinearActivationFunction;
using Winzent::ANN::SigmoidActivationFunction;
using Winzent::ANN::Individual;
using Winzent::ANN::REvolutionaryTrainingAlgorithm;
using Winzent::ANN::TrainingSet;
using Winzent::ANN::TrainingItem;
using Winzent::ANN::ValueVector;


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


void REvolutionaryTrainingAlgorithmTest::testIndividualOperatorEquals()
{
    NeuralNetwork *network = createNeuralNetwork();

    network->eachConnection([](Connection *const &c) {
        if (!c->fixedWeight()) {
            c->weight(0.0);
        }
    });

    Individual i1(network), i2(network);

    QVERIFY(i1 == i2);

    std::for_each(i2.parameters().begin(), i2.parameters().end(), [](qreal &w) {
        w = 1.0;
    });

    QVERIFY(!(i1 == i2));

    std::for_each(i2.parameters().begin(), i2.parameters().end(), [](qreal &w) {
        w = 0.0;
    });

    i1.scatter()[1] = 1.0;
    i2.scatter()[1] = 1.1;

    QVERIFY(!(i1 == i2));
    i2.scatter()[1] = 1.0;
    QVERIFY(i1 == i2);

    i1.errorVector()[0] = 11.1;
    QVERIFY(!(i1 == i2));
    i2.errorVector()[0] = 11.1;
    QVERIFY(i1 == i2);
}


void REvolutionaryTrainingAlgorithmTest::testIndividualOperatorAssign()
{
    Individual i1(createNeuralNetwork()), i2(createNeuralNetwork());

    if (i1 == i2) {
        i1.errorVector()[0] = 421.43;
        i2.errorVector()[0] = -21.43;
    }

    QVERIFY(!(i1 == i2));

    i1 = i2;

    QVERIFY(i1 == i2);
    QVERIFY(&i1 != &i2);
}


void REvolutionaryTrainingAlgorithmTest::testParametersSettingAndRetrieval()
{
    NeuralNetwork *neuralNetwork = createNeuralNetwork();
    Individual individual(neuralNetwork);

    ValueVector parameters = individual.parameters();
    QList<Connection *> connections;

    for (int i = 0; i != neuralNetwork->size(); ++i) {
        Layer *l = neuralNetwork->layerAt(i);

        for (int j = 0; j != l->size(); ++j) {
            Neuron *n = l->neuronAt(j);

            foreach (Connection *c, neuralNetwork->neuronConnectionsFrom(n)) {
                if (!c->fixedWeight()) {
                    connections << c;
                }
            }
        }
    }

    QList<Connection *> biasConnections =
            neuralNetwork->neuronConnectionsFrom(neuralNetwork->biasNeuron());
    foreach (Connection *c, biasConnections) {
        if (!c->fixedWeight()) {
            connections << c;
        }
    }

    QCOMPARE(parameters.size(), connections.size());

    parameters.clear();
    for (int i = 0; i != connections.size(); ++i) {
        parameters << 10.10;
    }

    individual.parameters(parameters);
    individual.applyParameters(neuralNetwork);

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


void REvolutionaryTrainingAlgorithmTest::testCompareIndividuals()
{
    Individual *i1 = new Individual(createNeuralNetwork());
    Individual *i2 = new Individual(createNeuralNetwork());

    QCOMPARE(0, i1->compare(i2));

    i2->timeToLive(3);
    QCOMPARE(0, i1->compare(i2));

    i2->age();
    QCOMPARE(0, i1->compare(i2));

    i1->errorVector()[0] = 1.0;
    QCOMPARE(1, i1->compare(i2));
    QVERIFY(i1->isBetterThan(i2));

    i2->errorVector()[0] = 1.0;
    QVERIFY(!i1->isBetterThan(i2));
    QCOMPARE(0, i2->compare(i1));

    i1->errorVector() << 1.0 << 2.0;
    i2->errorVector() << 1.0 << 1.0;
    QVERIFY(i2->isBetterThan(i1));

    delete i1;
    delete i2;
}


void REvolutionaryTrainingAlgorithmTest::testModifyIndividual()
{
    qsrand(time(NULL));

    REvolutionaryTrainingAlgorithm trainingAlgorithm(createNeuralNetwork());
    trainingAlgorithm.eliteSize(2).populationSize(5);
    TrainingSet trainingSet({ }, 1.0, 1000);

    QList<Individual *> population;
    for (int i = 0; i != trainingAlgorithm.populationSize(); ++i) {
        population << new Individual(createNeuralNetwork());
    }

    Individual *i3 = trainingAlgorithm.modifyIndividual(
            population.last(),
            population);

    QCOMPARE(i3->parameters().size(), population.first()->parameters().size());

    std::for_each(population.begin(), population.end() - 1, [&](Individual *i) {
        for (int j = 0; j != i->parameters().size(); ++j) {
            QVERIFY (i3->parameters().at(j) != i->parameters().at(j));
        }
    });

    foreach (Individual *i, population) {
        delete i;
    }
}


void REvolutionaryTrainingAlgorithmTest::testSortPopulation()
{
    Individual *i1 = new Individual(createNeuralNetwork());
    Individual *i2 = new Individual(createNeuralNetwork());
    Individual *i3 = new Individual(createNeuralNetwork());

    i1->timeToLive(10);
    i1->errorVector()[0] = 0.25;

    i2->timeToLive(10);
    i2->errorVector()[0] = 0.5;

    i3->timeToLive(2);
    i3->errorVector()[0] = 0.5;

    QVERIFY(i1->isBetterThan(i2));

    QList<Individual *> population = { i2, i1, i3 };

    QCOMPARE(population.first(), i2);
    REvolutionaryTrainingAlgorithm::sortPopulation(population);
    QCOMPARE(population, QList<Individual *>({ i1, i2, i3 }));
}


void REvolutionaryTrainingAlgorithmTest::testTrainXOR()
{
    qsrand(time(NULL));

    NeuralNetwork *network = new NeuralNetwork(this);
    PerceptronNetworkPattern *pattern = new PerceptronNetworkPattern(
            {
                2,
                3,
                1
            }, {
                new LinearActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            },
            this);
    network->configure(pattern);

    // Build training data:

    QList<TrainingItem> trainingItems;
    trainingItems
            << TrainingItem({ 0.0, 0.0 }, { 0.0 })
            << TrainingItem({ 0.0, 1.0 }, { 1.0 })
            << TrainingItem({ 1.0, 0.0 }, { 1.0 })
            << TrainingItem({ 1.0, 1.0 }, { 0.0 });

    TrainingSet *trainingSet = new TrainingSet(
            trainingItems,
            1e-3,
            15000);

    REvolutionaryTrainingAlgorithm trainingAlgorithm(network);
    trainingAlgorithm
            .populationSize(50)
            .eliteSize(5)
            .maxNoSuccessEpochs(INT_MAX)
            .startTTL(500)
            .gradientWeight(3.0)
            .ebmin(1e-2)
            .ebmax(2.0)
            .successWeight(0.1);

    QDateTime dt1 = QDateTime::currentDateTime();
    trainingAlgorithm.train(trainingSet);
    QDateTime dt2 = QDateTime::currentDateTime();

    qDebug() << "Trained XOR(x, y) in" << dt1.msecsTo(dt2) << "msec";

    ValueVector output;
    output = network->calculate({ 1, 1 });
    qDebug() << "(1, 1) =>" << output;
    QCOMPARE(qRound(output[0]), 0);
    output = network->calculate({ 1, 0 });
    qDebug() << "(1, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 1);
    output = network->calculate({ 0, 0 });
    qDebug() << "(0, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 0);
    output = network->calculate({ 0, 1 });
    qDebug() << "(0, 1) =>" << output;
    QCOMPARE(qRound(output[0]), 1);

    delete trainingSet;
}


TESTCASE(REvolutionaryTrainingAlgorithmTest)
