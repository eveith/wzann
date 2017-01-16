#include <cstddef>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "SigmoidActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "NguyenWidrowWeightRandomizer.h"

#include <gtest/gtest.h>

#include "NguyenWidrowWeightRandomizerTest.h"


using namespace Winzent::ANN;



TEST(NguyenWidrowWeightRandomizerTest, testRandomizeWeights)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern({
            1,
            2,
            3
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });
    network.configure(pattern);

    NguyenWidrowWeightRandomizer().randomize(network);

    for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
        Layer &layer = network[i];

        for (Layer::size_type j = 0; j != layer.size(); ++j) {
            Neuron &neuron = layer[j];
            for (const auto &c: boost::make_iterator_range(
                     network.connectionsFrom(neuron))) {
                ASSERT_TRUE(1.0 != 1.0 + c->weight());
            }
        }

        for (const auto &c: boost::make_iterator_range(
                 network.connectionsFrom(network.biasNeuron()))) {
            ASSERT_EQ(-1.0, c->weight());
        }
    }
}
