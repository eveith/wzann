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

#include "PsoTrainingAlgorithm.h"
#include "PsoTrainingAlgorithmTest.h"


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
using Winzent::ANN::PsoTrainingAlgorithm;


PsoTrainingAlgorithmTest::PsoTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


NeuralNetwork *PsoTrainingAlgorithmTest::createNeuralNetwork()
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


void PsoTrainingAlgorithmTest::testTrainXOR()
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

    PsoTrainingAlgorithm trainingAlgorithm;
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


TESTCASE(PsoTrainingAlgorithmTest)
