#include "Testrunner.h"

#include "ActivationFunctionTest.h"

#include "SigmoidActivationFunction.h"
#include "LinearActivationFunction.h"



using namespace Winzent::ANN;


ActivationFunctionTest::ActivationFunctionTest(QObject *parent):
        QObject(parent)
{
}


void ActivationFunctionTest::testSigmoidActivationFunction()
{
    SigmoidActivationFunction a;

    QVERIFY(a.hasDerivative());
    QCOMPARE(a.calculate(8.0), 1 / (1 + std::exp(-8.0)));
    QCOMPARE(1.0 + a.calculate(0.0), 1.5);
    QCOMPARE(
            a.calculateDerivative(5.0, 0.5),
            0.5 * (1.0 - 0.5));
}


void ActivationFunctionTest::testLinarActivationFunction()
{
    LinearActivationFunction a;

    QVERIFY(a.hasDerivative());
    QCOMPARE(a.calculate(1.5), 1.5);
    QCOMPARE(a.calculate(-2.75), -2.75);

    LinearActivationFunction b(2.0);
    QCOMPARE(b.calculate(1.5), 3.0);
    QCOMPARE(b.calculate(-2.75), -5.5);
}


TESTCASE(ActivationFunctionTest)
