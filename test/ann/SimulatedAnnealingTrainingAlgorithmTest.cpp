#include <QObject>
#include <QtTest>
#include <QList>
#include <QFile>
#include <QTextStream>

#include "NeuralNetwork.h"

#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"

#include "PerceptronNetworkPattern.h"

#include "TrainingSet.h"
#include "SimulatedAnnealingTrainingAlgorithm.h"

#include "Testrunner.h"
#include "SimulatedAnnealingTrainingAlgorithmTest.h"


using namespace Winzent::ANN;


SimulatedAnnealingTrainingAlgorithmTest::SimulatedAnnealingTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


void SimulatedAnnealingTrainingAlgorithmTest::testTrainXOR()
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
            << TrainingItem(ValueVector({ 0.0, 0.0 }), ValueVector({ 0.0 }))
            << TrainingItem(ValueVector({ 0.0, 1.0 }), ValueVector({ 1.0 }))
            << TrainingItem(ValueVector({ 1.0, 0.0 }), ValueVector({ 1.0 }))
            << TrainingItem(ValueVector({ 1.0, 1.0 }), ValueVector({ 0.0 }));

    TrainingSet *trainingSet = new TrainingSet(
            trainingItems,
            0.7,
            0.1,
            100);

    QFile testResultFile(QString(QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);

    testResultStream << *network;

    network->train(
            new SimulatedAnnealingTrainingAlgorithm(10, 2, 100, this),
            trainingSet);

    testResultStream << *network;

    ValueVector output;
    output = network->calculate(ValueVector({ 1, 1 }));
    qDebug() << "(1, 1) =>" << output;
    QCOMPARE(qRound(output[0]), 0);
    output = network->calculate(ValueVector({ 1, 0 }));
    qDebug() << "(1, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 1);
    output = network->calculate(ValueVector({ 0, 0 }));
    qDebug() << "(0, 0) =>" << output;
    QCOMPARE(qRound(output[0]), 0);
    output = network->calculate(ValueVector({ 0, 1 }));
    qDebug() << "(0, 1) =>" << output;
    QCOMPARE(qRound(output[0]), 1);


    testResultStream.flush();
    testResultFile.flush();
    testResultFile.close();
}


TESTCASE(SimulatedAnnealingTrainingAlgorithmTest);
