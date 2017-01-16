#include <gtest/gtest.h>

#include "Neuron.h"
#include "LinearActivationFunction.h"

#include "Layer.h"
#include "LayerTest.h"


using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;
using Winzent::ANN::LinearActivationFunction;


TEST(LayerTest, testLayerCreation)
{
    Layer layer;
    ASSERT_EQ(0ul, layer.size());
}


TEST(LayerTest, testNeuronAddition)
{
    Layer layer;
    layer << new Neuron(new LinearActivationFunction());
    layer << new Neuron(new LinearActivationFunction());

    ASSERT_EQ(2ul, layer.size());
    ASSERT_TRUE(layer.neuronAt(0)->parent() == &layer);
    ASSERT_TRUE(layer.neuronAt(1)->parent() == &layer);

    for (const Neuron &n: layer) {
        ASSERT_TRUE(layer.contains(n));
    }
}


TEST(LayerTest, testNeuronIterator)
{
    QList<const Neuron*> neurons;

    Layer layer;
    layer << new Neuron(new LinearActivationFunction());
    layer << new Neuron(new LinearActivationFunction());

    for (const Neuron &neuron: layer) {
        neurons << &neuron;
    }

    ASSERT_EQ(layer.size(, static_cast<size_t>(neurons.size())));
}
