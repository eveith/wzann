#include <gtest/gtest.h>

#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "SimpleWeightRandomizer.h"
#include "SimpleWeightRandomizerTest.h"


using boost::make_iterator_range;

using wzann::Connection;
using wzann::NeuralNetwork;
using wzann::ActivationFunction;
using wzann::SimpleWeightRandomizer;
using wzann::PerceptronNetworkPattern;


TEST(SimpleWeightRandomizerTest, testWeightRandomization)
{
    NeuralNetwork neuralNetwork;
    PerceptronNetworkPattern pattern;

    pattern.addLayer({ 5, ActivationFunction::Identity });
    pattern.addLayer({ 20, ActivationFunction::Identity });
    neuralNetwork.configure(pattern);

    for (auto const* c: make_iterator_range(neuralNetwork.connections())) {
        if (&(c->source()) != &(neuralNetwork.biasNeuron())) {
            ASSERT_EQ(c->weight() + 1.0, 1.0);
        }
    }

    SimpleWeightRandomizer swr;
    swr.randomize(neuralNetwork);

    for (auto const* c: make_iterator_range(neuralNetwork.connections())) {
        if (&(c->source()) != &(neuralNetwork.biasNeuron())) {
            ASSERT_TRUE(1.0 != c->weight() + 1.0);
            ASSERT_TRUE(swr.minWeight() <= c->weight());
            ASSERT_TRUE(c->weight() <= swr.maxWeight());
        }
    }
}
