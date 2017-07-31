#include <gtest/gtest.h>

#include <list>

#include "Neuron.h"
#include "ActivationFunction.h"

#include "Layer.h"
#include "LayerTest.h"


using namespace wzann;


TEST(LayerTest, testLayerCreation)
{
    Layer layer;
    ASSERT_EQ(0ul, layer.size());
}


TEST(LayerTest, testNeuronAddition)
{
    Layer layer;
    layer << new Neuron();
    layer << new Neuron();

    ASSERT_EQ(2ul, layer.size());
    ASSERT_TRUE(layer.neuronAt(0)->parent() == &layer);
    ASSERT_TRUE(layer.neuronAt(1)->parent() == &layer);

    for (auto const& n: layer) {
        ASSERT_TRUE(layer.contains(n));
    }
}


TEST(LayerTest, testNeuronIterator)
{
    std::vector<Neuron const*> neurons;

    Layer layer;
    layer << new Neuron();
    layer << new Neuron();

    for (auto const& neuron: layer) {
        neurons.push_back(&neuron);
    }

    ASSERT_EQ(
            layer.size(),
            neurons.size());
}


TEST(LayerTest, testSerialization)
{
    Layer layer1;

    auto* n1 = new Neuron(), *n2 = new Neuron();
    n1->activationFunction(ActivationFunction::Logistic);
    n2->activationFunction(ActivationFunction::Logistic);

    layer1 << n1 << n2;

    auto v = to_variant(layer1);
    auto* layer2 = new_from_variant<Layer>(v);

    ASSERT_NE(&layer1, layer2);
    ASSERT_EQ(layer1, *layer2);

    delete layer2;
}
