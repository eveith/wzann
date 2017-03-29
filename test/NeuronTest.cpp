#include <gtest/gtest.h>

#include <Variant/Variant.h>

#include "Neuron.h"
#include "NeuronTest.h"


using namespace Winzent::ANN;


TEST(NeuronTest, testClone)
{
    Neuron n1;
    n1.activationFunction(ActivationFunction::ReLU);

    n1.activate(42.);

    Neuron *n2 = n1.clone();

    ASSERT_EQ(n1, *n2);
    ASSERT_NE(&n1, n2);

    delete n2;
}


TEST(NeuronTest, testSerialize)
{
    Neuron n1;
    n1.activationFunction(ActivationFunction::Gaussian);
    n1.activate(2.4);

    auto v = to_variant(n1);
    auto* n2 = new_from_variant<Neuron>(v);

    ASSERT_EQ(n1, *n2);
    ASSERT_NE(&n1, n2);

    delete n2;
}
