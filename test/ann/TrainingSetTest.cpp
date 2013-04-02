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
    QCOMPARE(TrainingItem(ValueVector(), ValueVector()).outputRelevant(), true);
    QCOMPARE(TrainingItem(ValueVector()).outputRelevant(), false);
}


TESTCASE(TrainingSetTest);
