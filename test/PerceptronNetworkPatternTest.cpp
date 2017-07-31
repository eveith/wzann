#include <gtest/gtest.h>

#include "Layer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"

#include "PerceptronNetworkPattern.h"
#include "PerceptronNetworkPatternTest.h"


using namespace wzann;


TEST(PerceptronNetworkPatternTest, testConfigure)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern;

    pattern.addLayer({ 2, ActivationFunction::Logistic });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    pattern.addLayer({ 1, ActivationFunction::Logistic });
    network.configure(pattern);

    for (size_t i = 0; i != network.size() - 1; ++i) {
        for (size_t j = 0; j != network.layerAt(i)->size(); ++j) {
            for (size_t k = 0; k != network[i+1].size(); ++k) {
                ASSERT_TRUE(network.connectionExists(
                    network[i][j],
                    network[i+1][k]));
                ASSERT_TRUE(network.connectionExists(
                    network.biasNeuron(),
                    network[i+1][k]));
            }
        }
    }

    ASSERT_FALSE(network.connectionExists(
            network[1][0],
            network[0][0]));
    ASSERT_FALSE(network.connectionExists(
            network[1][0],
            network[1][0]));
    ASSERT_FALSE(network.connectionExists(
            network[1][0],
            network[1][1]));
}


TEST(PerceptronNetworkPatternTest, testCalculate)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern;

    pattern.addLayer({ 2, ActivationFunction::Logistic });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    pattern.addLayer({ 1, ActivationFunction::Logistic });
    network.configure(pattern);

    Vector input = { 1.0, 0.0 };
    Vector output = network.calculate(input);
    ASSERT_TRUE(1.0f != output[0] + 1.0);
}
