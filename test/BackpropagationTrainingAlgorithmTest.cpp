#include <iostream>

#include <gtest/gtest.h>

#include "TrainingSet.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "SimpleWeightRandomizer.h"
#include "PerceptronNetworkPattern.h"


#include "BackpropagationTrainingAlgorithm.h"
#include "BackpropagationTrainingAlgorithmTest.h"


using namespace wzann;


TEST(BackpropagationTrainingAlgorithmTest, testTrainXOR)
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern;
    pattern.addLayer({ 2, ActivationFunction::Identity });
    pattern.addLayer({ 3, ActivationFunction::Logistic });
    pattern.addLayer({ 1, ActivationFunction::Logistic });
    network.configure(pattern);
    SimpleWeightRandomizer().randomize(network);

    // Build training data:

    double targetVariance = 1e-2;
    double targetTrainingError = targetVariance * targetVariance / 4. * 0.5;

    TrainingSet trainingSet;
    trainingSet.targetError(targetTrainingError).maxEpochs(100000)
            << TrainingItem({ 0.0, 0.0 }, { 0.0 })
            << TrainingItem({ 0.0, 1.0 }, { 1.0 })
            << TrainingItem({ 1.0, 0.0 }, { 1.0 })
            << TrainingItem({ 1.0, 1.0 }, { 0.0 });
    BackpropagationTrainingAlgorithm()
            .learningRate(1.0)
            .train(network, trainingSet);

    std::cout << "Error: " << trainingSet.error()
            << ", Epochs: " << trainingSet.epochs() << "\n";

    Vector output;
    output = network.calculate({ 1., 1. });
    ASSERT_NEAR(0., output[0], targetVariance);
    output = network.calculate({ 1, 0 });
    ASSERT_NEAR(1., output[0], targetVariance);
    output = network.calculate({ 0, 0 });
    ASSERT_NEAR(0., output[0], targetVariance);
    output = network.calculate({ 0, 1 });
    ASSERT_NEAR(1., output[0], targetVariance);

    ASSERT_LE(trainingSet.error(), targetTrainingError);
    ASSERT_TRUE(trainingSet.epochs() < trainingSet.maxEpochs());
}
