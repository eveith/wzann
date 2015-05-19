#include <QtTest>
#include <QObject>

#include "NeuralNetwork.h"
#include "SimpleWeightRandomizer.h"
#include "PerceptronNetworkPattern.h"

#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"

#include "TrainingSet.h"

#include "RpropTrainingAlgorithm.h"

#include "Testrunner.h"
#include "RpropTrainingAlgorithmTest.h"


using namespace Winzent::ANN;


RpropTrainingAlgorithmTest::RpropTrainingAlgorithmTest(QObject *parent):
        QObject(parent)
{
}


void RpropTrainingAlgorithmTest::testTrainXOR()
{
    NeuralNetwork *network = new NeuralNetwork(this);
    PerceptronNetworkPattern pattern(
            { 2, 3, 1 },
            {
                new LinearActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });

    network->configure(pattern);
    SimpleWeightRandomizer().randomize(*network);

    TrainingSet ts(
            {
                TrainingItem({ 1.0, 1.0 }, { 0.0 }),
                TrainingItem({ 1.0, 0.0 }, { 1.0 }),
                TrainingItem({ 0.0, 0.0 }, { 0.0 }),
                TrainingItem({ 0.0, 1.0 }, { 1.0 })
            },
            1e-3,
            2000);
    RpropTrainingAlgorithm trainingAlgorithm;
    trainingAlgorithm.train(network, ts);

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


TESTCASE(RpropTrainingAlgorithmTest)
