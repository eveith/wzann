#include "Testrunner.h"

#define private public
#include "Neuron.h"
#include "SigmoidActivationFunction.h"

#include "NeuronTest.h"


using namespace Winzent::ANN;


void NeuronTest::testClone()
{
    Neuron* n1 = new Neuron(new SigmoidActivationFunction(1.0, this));
    Neuron* n2 = n1->clone();

    QVERIFY(n1 != n2);
    QVERIFY(n1->m_activationFunction != n2->m_activationFunction);

    delete n1;
    QVERIFY(n2->activate(0.5));
}


TESTCASE(NeuronTest)
