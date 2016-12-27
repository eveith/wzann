#include <gtest/gtest.h>

#include "Neuron.h"
#include "SigmoidActivationFunction.h"

#include "NeuronTest.h"


using namespace Winzent::ANN;


TEST(NeuronTest, testClone)
{
    Neuron n1(new SigmoidActivationFunction());
    Neuron *n2 = n1.clone();

    ASSERT_TRUE(&n1 != n2);
    ASSERT_TRUE(n1.activationFunction() != n2->activationFunction());

    ASSERT_TRUE(n2->activate(0.5));
    delete n2;
}
