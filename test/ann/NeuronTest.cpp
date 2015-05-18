#include "Testrunner.h"

#define private public
#include "Neuron.h"
#include "SigmoidActivationFunction.h"

#include "NeuronTest.h"


using namespace Winzent::ANN;


void NeuronTest::testClone()
{
    Neuron n1(new SigmoidActivationFunction());
    Neuron *n2 = n1.clone();

    QVERIFY(&n1 != n2);
    QVERIFY(n1.activationFunction() != n2->activationFunction());

    QVERIFY(n2->activate(0.5));
    delete n2;
}


TESTCASE(NeuronTest)
