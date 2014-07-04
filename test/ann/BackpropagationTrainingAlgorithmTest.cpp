#include <initializer_list>

#include "NeuralNetwork.h"
#include "PerceptronNetworkPattern.h"
#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"
#include "TrainingSet.h"
#include "BackpropagationTrainingAlgorithm.h"

#include "Testrunner.h"
#include "BackpropagationTrainingAlgorithmTest.h"


using namespace Winzent::ANN;


BackpropagationTrainingAlgorithmTest::BackpropagationTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


void BackpropagationTrainingAlgorithmTest::testTrainXOR()
{
    qsrand(time(NULL));

    NeuralNetwork *network = new NeuralNetwork(this);
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

    network->configure(pattern);

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
            50000);

    QFile testResultFile(QString(QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);

    testResultStream << *network;

    BackpropagationTrainingAlgorithm().train(network, trainingSet);

    testResultStream << *network;

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


    testResultStream.flush();
    testResultFile.flush();
    testResultFile.close();
}


TESTCASE(BackpropagationTrainingAlgorithmTest)
