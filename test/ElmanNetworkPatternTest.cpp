#include <gtest/gtest.h>

#include <QTextStream>

#include <initializer_list>

#include "ElmanNetworkPatternTest.h"

#include "ElmanNetworkPattern.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Connection;
using Winzent::ANN::ElmanNetworkPattern;
using Winzent::ANN::ActivationFunction;
using Winzent::ANN::SigmoidActivationFunction;


ElmanNetworkPatternTest::ElmanNetworkPatternTest
{
    layers
            << 1
            << 10
            << 1;
    activationFunctions
            << new SigmoidActivationFunction()
            << new SigmoidActivationFunction()
            << new SigmoidActivationFunction();
}


TEST_F(ElmanNetworkPatternTest, testConfigure)
{
    NeuralNetwork network;
    ElmanNetworkPattern pattern(layers, activationFunctions);

    network.configure(pattern);

    ASSERT_EQ(4ul, network.size());

    // Each hidden layer neuron has one context neuron: Check

    for (int i = 1; i != layers.at(1); ++i) {
        for (int j = 1; j != layers.at(1); ++j) {
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
