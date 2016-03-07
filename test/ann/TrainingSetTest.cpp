#include <QtTest>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "Testrunner.h"
#include "TrainingSetTest.h"


using namespace Winzent::ANN;


TrainingSetTest::TrainingSetTest(QObject *parent) :
    QObject(parent)
{
}


void TrainingSetTest::testOutputRelevant()
{
    QCOMPARE(
            TrainingItem(Vector({ 1.0 }), Vector({ 1.0 })).outputRelevant(),
            true);
    QCOMPARE(
            TrainingItem(Vector()).outputRelevant(),
            false);
}


void TrainingSetTest::testJsonSerialization()
{
    TrainingSet ts;
    ts.targetError(0.01);
    ts.maxEpochs(1000);
    ts
            << TrainingItem({ 1.0, 1.0, 1.0 })
            << TrainingItem({ 2.0, 2.0, 2.0}, { 3.0, 3.0, 3.0 });

    auto json = ts.toJSON();
    qDebug() << json;

    TrainingSet ts2;
    Winzent::deserialize(ts2, json);

    QCOMPARE(ts2.error(), ts.error());
    QCOMPARE(ts2.maxEpochs(), ts.maxEpochs());
    QCOMPARE(ts2.trainingItems.size(), ts.trainingItems.size());
}


TESTCASE(TrainingSetTest);
