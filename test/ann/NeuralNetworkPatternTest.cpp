/*!
 * \file	NeuralNetworkPatternTest.cpp
 * \brief
 * \date	03.01.2013
 * \author	eveith
 */


#include "Testrunner.h"

#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "SigmoidActivationFunction.h"
#include "NeuralNetworkPatternTest.h"


using Mock::NeuralNetworkPatternTestDummyPattern;

using namespace Winzent::ANN;


NeuralNetworkPatternTestDummyPattern::NeuralNetworkPatternTestDummyPattern():
        NeuralNetworkPattern(QList<int>(), QList<ActivationFunction*>()),
        numLayers(3), numNeuronsPerLayer(10)
{
    for (int i = 0; i != numLayers; ++i) {
        m_layerSizes << numNeuronsPerLayer;
        m_activationFunctions << new SigmoidActivationFunction();
    }
}


ValueVector NeuralNetworkPatternTestDummyPattern::calculate(
        NeuralNetwork*, const ValueVector& input)
{
    return input;
}


NeuralNetworkPattern* NeuralNetworkPatternTestDummyPattern::clone() const
{
    return new NeuralNetworkPatternTestDummyPattern();
}


void NeuralNetworkPatternTestDummyPattern::configureNetwork(
        NeuralNetwork* network)
{
    for (int i = 0; i != numLayers; ++i) {
        Layer *l = new Layer(network);

        for (int j = 0; j != numNeuronsPerLayer; ++j) {
            Neuron *n = new Neuron(
                    new SigmoidActivationFunction(),
                    network);
            l->neurons << n;
        }

        *network << l;

        if (i > 0) {
            fullyConnectNetworkLayers(network, i-1, i);
        }
    }
}


void NeuralNetworkPatternTest::testFullyConnectNetworkLayers()
{
    NeuralNetwork network;
    Mock::NeuralNetworkPatternTestDummyPattern pattern;

    network.configure(&pattern);

    // Check fully connected state:

    for (int i = 1; i != pattern.numLayers - 1; ++i) {
        for (int j = 1; j != pattern.numNeuronsPerLayer; ++j) {
            QVERIFY2(network.neuronConnectionExists(
                        network.layerAt(i)->neuronAt(j),
                        network.layerAt(i+1)->neuronAt(j)),
                    "All neurons must be connected to each other.");
        }
    }
}


TESTCASE(NeuralNetworkPatternTest)
