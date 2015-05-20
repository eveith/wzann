#include <QObject>
#include <QtTest>
#include <QList>
#include <QFile>
#include <QTextStream>

#include "NeuralNetwork.h"

#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"

#include "PerceptronNetworkPattern.h"
#include <NguyenWidrowWeightRandomizer.h>

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
    NguyenWidrowWeightRandomizer().randomize(network);

    // Build training data:

    QList<TrainingItem> trainingItems;
    trainingItems
            << TrainingItem({ 0.0, 0.0 }, { 0.0 })
            << TrainingItem({ 0.0, 1.0 }, { 1.0 })
            << TrainingItem({ 1.0, 0.0 }, { 1.0 })
            << TrainingItem({ 1.0, 1.0 }, { 0.0 });

    TrainingSet trainingSet(
            trainingItems,
            1e-3,
            10000);

    QFile testResultFile(QString(QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);

    testResultStream << network;

    QDateTime dt1 = QDateTime::currentDateTime();

    SimulatedAnnealingTrainingAlgorithm(10, 2, 100)
        .train(&network, trainingSet);

    QDateTime dt2 = QDateTime::currentDateTime();
    qDebug() << "Trained XOR(x, y) in" << dt1.msecsTo(dt2) << "msec";

    testResultStream << network;

    ValueVector output;
    output = network.calculate({ 1, 1 });
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


    testResultStream.flush();
    testResultFile.flush();
    testResultFile.close();
}


TESTCASE(SimulatedAnnealingTrainingAlgorithmTest)
