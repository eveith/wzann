#include <QtTest>
#include <QObject>

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

#include "PsoTrainingAlgorithm.h"
#include "PsoTrainingAlgorithmTest.h"


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
using wzann::PsoTrainingAlgorithm;


NeuralNetwork* PsoTrainingAlgorithmTest::createNeuralNetwork()
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


TEST_F(PsoTrainingAlgorithmTest, testTrainXOR)
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
