#include "Testrunner.h"

#include "ActivationFunctionTest.h"

#define protected public
#include "SigmoidActivationFunction.h"
#include "LinearActivationFunction.h"



using namespace Winzent::ANN;


ActivationFunctionTest::ActivationFunctionTest(QObject *parent) :
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
            a.calculateDerivative(1.0),
            a.calculate(1.0) * (1.0 - a.calculate(1.0)));

    SigmoidActivationFunction b(1.0, -0.5);
    QCOMPARE(1.0 + b.calculate(0.0), 1.0);
    QCOMPARE(b.calculateDerivative(0.0), -0.5);

    SigmoidActivationFunction *c =
            new SigmoidActivationFunction(2.0, -1.0, this);
    QCOMPARE(1.0 + c->calculate(0.0), 1.0);

    SigmoidActivationFunction *d =
            static_cast<SigmoidActivationFunction*>(c->clone());
    QCOMPARE(d->m_scalingFactor, c->m_scalingFactor);
    QCOMPARE(d->m_transposition, c->m_transposition);
}


void ActivationFunctionTest::testLinarActivationFunction()
{
    LinearActivationFunction a;

    QVERIFY(a.hasDerivative());
    QCOMPARE(a.calculate(1.5), 1.5);
    QCOMPARE(a.calculate(-2.75), -2.75);

    LinearActivationFunction b(2.0, 4.0);
    QCOMPARE(b.calculate(1.5), 7.0);
    QCOMPARE(b.calculate(-2.75), -1.5);
}


TESTCASE(ActivationFunctionTest)
