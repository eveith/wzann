#include <gtest/gtest.h>

#include "Layer.h"
#include "Neuron.h"
#include "NeuralNetwork.h"

#include "NeuralNetworkPattern.h"
#include "NeuralNetworkPatternTest.h"


using Mock::NeuralNetworkPatternTestDummyPattern;

using namespace wzann;


NeuralNetworkPatternTestDummyPattern::NeuralNetworkPatternTestDummyPattern():
        NeuralNetworkPattern(),
        numLayers(3),
        numNeuronsPerLayer(10)
{
    for (size_t i = 0; i != numLayers; ++i) {
        addLayer(SimpleLayerDefinition(
                numNeuronsPerLayer,
                ActivationFunction::Identity));
    }
}


Vector NeuralNetworkPatternTestDummyPattern::calculate(
        NeuralNetwork &, Vector const& input)
{
    return input;
}


NeuralNetworkPattern* NeuralNetworkPatternTestDummyPattern::clone() const
{
    return new NeuralNetworkPatternTestDummyPattern();
}


void NeuralNetworkPatternTestDummyPattern::configureNetwork(
        NeuralNetwork &network)
{
    for (size_t i = 0; i != numLayers; ++i) {
        auto* l = new Layer();

        for (size_t j = 0; j != numNeuronsPerLayer; ++j) {
            auto* n = new Neuron();
            n->activationFunction(ActivationFunction::Identity);
            *l << n;
        }

        network << l;

        if (i > 0) {
            fullyConnectNetworkLayers(network[i-1], network[i]);
        }
    }
}


TEST(NeuralNetworkPatternTest, testFullyConnectNetworkLayers)
{
    NeuralNetwork network;
    Mock::NeuralNetworkPatternTestDummyPattern pattern;

    network.configure(pattern);

    // Check fully connected state:

    for (size_t i = 1; i != pattern.numLayers - 1; ++i) {
        for (size_t j = 1; j != pattern.numNeuronsPerLayer; ++j) {
            ASSERT_TRUE(network.connectionExists(
                    network[i][j],
                    network[i+1][j]));
        }
    }
}
