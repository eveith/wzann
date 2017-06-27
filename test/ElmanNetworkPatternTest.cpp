#include <gtest/gtest.h>


#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ActivationFunction.h"

#include "ElmanNetworkPattern.h"
#include "ElmanNetworkPatternTest.h"


using namespace Winzent::ANN;


ElmanNetworkPatternTest::ElmanNetworkPatternTest()
{
    m_layers.push_back({ 1, ActivationFunction::ReLU });
    m_layers.push_back({ 10, ActivationFunction::Logistic });
    m_layers.push_back({ 1, ActivationFunction::Logistic });
}


TEST_F(ElmanNetworkPatternTest, testConfigure)
{
    NeuralNetwork network;
    ElmanNetworkPattern pattern;

    for (auto const& layerDefinition : m_layers) {
        pattern.addLayer(layerDefinition);
    }

    network.configure(pattern);

    ASSERT_EQ(4ul, network.size());

    // Each hidden layer neuron has one context neuron: Check

    for (Layer::size_type i = 1; i != m_layers[1].first; ++i) {
        for (Layer::size_type j = 1; j != m_layers[1].first; ++j) {
            bool connection = network.connectionExists(
                    network[ElmanNetworkPattern::HIDDEN][i],
                    network[ElmanNetworkPattern::CONTEXT][j]);

            if (i == j) {
                ASSERT_EQ(true, connection);
                Connection *c = network.connection(
                        network[ElmanNetworkPattern::HIDDEN][i],
                        network[ElmanNetworkPattern::CONTEXT][j]);
                ASSERT_EQ(1.0, c->weight());
                ASSERT_EQ(true, c->fixedWeight());
            } else {
                ASSERT_EQ(false, connection);
            }
        }
    }
}
