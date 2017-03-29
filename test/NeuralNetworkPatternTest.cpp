#include <gtest/gtest.h>

#include "Layer.h"
#include "Neuron.h"
#include "NeuralNetwork.h"
#include "SigmoidActivationFunction.h"

#include "NeuralNetworkPattern.h"
#include "NeuralNetworkPatternTest.h"


using Mock::NeuralNetworkPatternTestDummyPattern;

using namespace Winzent::ANN;


NeuralNetworkPatternTestDummyPattern::NeuralNetworkPatternTestDummyPattern():
        NeuralNetworkPattern(QList<int>(), QList<ActivationFunction *>()),
        numLayers(3), numNeuronsPerLayer(10)
{
    for (int i = 0; i != numLayers; ++i) {
        m_layerSizes << numNeuronsPerLayer;
        m_activationFunctions << new SigmoidActivationFunction();
    }
}


Vector NeuralNetworkPatternTestDummyPattern::calculate(
        NeuralNetwork &, const Vector &input)
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
    for (int i = 0; i != numLayers; ++i) {
        Layer *l = new Layer();

        for (int j = 0; j != numNeuronsPerLayer; ++j) {
            Neuron *n = new Neuron(new SigmoidActivationFunction());
            *l << n;
        }

        network << l;

        if (i > 0) {
            fullyConnectNetworkLayers(network, i-1, i);
        }
    }
}


TEST(NeuralNetworkPatternTest, testFullyConnectNetworkLayers)
{
    NeuralNetwork network;
    Mock::NeuralNetworkPatternTestDummyPattern pattern;

    network.configure(pattern);

    // Check fully connected state:

    for (int i = 1; i != pattern.numLayers - 1; ++i) {
        for (int j = 1; j != pattern.numNeuronsPerLayer; ++j) {
            ASSERT_TRUE2(network.connectionExists(
                        network[i][j],
                        network[i+1][j]),
                    "All neurons must be connected to each other.");
        }
    }
}