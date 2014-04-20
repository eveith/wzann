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
#include "NguyenWidrowWeightRandomizer.h"
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
using Winzent::ANN::NguyenWidrowWeightRandomizer;
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

    ValueVector parameters = individual.parameters();
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


void REvolutionaryTrainingAlgorithmTest::testCompareIndividuals()
{
    Individual *i1 = new Individual(createNeuralNetwork());
    Individual *i2 = new Individual(createNeuralNetwork());

    QCOMPARE(0, i1->compare(i2));

    i2->timeToLive(3);
    QCOMPARE(-1, i1->compare(i2));

    i1->timeToLive(3);
    QCOMPARE(0, i2->compare(i1));

    i2->age();
    QCOMPARE(1, i1->compare(i2));
    QVERIFY(i1->isBetterThan(i2));

    i1->age();
    i1->errorVector()[0] = 1.0;
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


void REvolutionaryTrainingAlgorithmTest::testGenerateIndividual()
{
    qsrand(time(NULL));

    REvolutionaryTrainingAlgorithm trainingAlgorithm(createNeuralNetwork());
    trainingAlgorithm.eliteSize(2).populationSize(5);
    TrainingSet trainingSet({ }, 1.0, 1000);

    QList<Individual *> population;
    for (int i = 0; i != trainingAlgorithm.populationSize(); ++i) {
        population << new Individual(createNeuralNetwork());
    }

    Individual *i3 = trainingAlgorithm.generateIndividual(
            population,
            &trainingSet);

    QCOMPARE(i3->parameters().size(), population.first()->parameters().size());

    foreach (Individual *i, population) {
        for (int j = 0; j != i->parameters().size(); ++j) {
            QVERIFY (i3->parameters().at(j) != i->parameters().at(j));
        }
    }

    delete i3;
    foreach (Individual *i, population) {
        delete i;
    }
}


void REvolutionaryTrainingAlgorithmTest::testSortPopulation()
{
    Individual *i1 = new Individual(createNeuralNetwork());
    Individual *i2 = new Individual(createNeuralNetwork());

    i1->timeToLive(10);
    i1->errorVector()[0] = 0.25;

    i2->timeToLive(2);
    i2->errorVector()[0] = 0.5;

    QVERIFY(i1->isBetterThan(i2));

    QList<Individual *> population = { i2, i1 };
    REvolutionaryTrainingAlgorithm::sortPopulation(population);

    QCOMPARE(population.first(), i1);
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
            0.1,
            5000);

    REvolutionaryTrainingAlgorithm trainingAlgorithm(network);
    trainingAlgorithm
            .populationSize(50)
            .eliteSize(5)
            .maxNoSuccessEpochs(INT_MAX)
            .measurementEpochs(1000)
            .startTTL(500)
            .gradientWeight(3.0)
            .train(trainingSet);

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

}


TESTCASE(REvolutionaryTrainingAlgorithmTest)
