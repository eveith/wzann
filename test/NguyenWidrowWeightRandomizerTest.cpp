#include <gtest/gtest.h>

#include <cstddef>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "NguyenWidrowWeightRandomizer.h"
#include "NguyenWidrowWeightRandomizerTest.h"


using namespace Winzent::ANN;



TEST(NguyenWidrowWeightRandomizerTest, testRandomizeWeights)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern;

    pattern.addLayer({ 1, ActivationFunction::Identity });
    pattern.addLayer({ 2, ActivationFunction::Logistic });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    network.configure(pattern);

    NguyenWidrowWeightRandomizer().randomize(network);

    for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
        auto& layer = network[i];

        for (Layer::size_type j = 0; j != layer.size(); ++j) {
            auto& neuron = layer[j];
            for (auto const& c: boost::make_iterator_range(
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
