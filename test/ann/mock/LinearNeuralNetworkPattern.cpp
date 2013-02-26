#include <QtTest>

#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"
#include "LinearActivationFunction.h"
#include "NeuralNetwork.h"

#include "LinearNeuralNetworkPattern.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::NeuralNetworkPattern;
using Winzent::ANN::LinearActivationFunction;


namespace Mock {
    
    LinearNeuralNetworkPattern::LinearNeuralNetworkPattern():
            NeuralNetworkPattern()
    {
        m_layerSizes << 2 << 6 << 2;
        m_activationFunctions <<
                new LinearActivationFunction(),
                new LinearActivationFunction(),
                new LinearActivationFunction();

        // Make sure we get constant weights:

        m_weightRandomMin = 1.0;
        m_weightRandomMax = 1.0;
    }


    void LinearNeuralNetworkPattern::configureNetwork(NeuralNetwork *network)
    {
    }
    
} // namespace Mock
