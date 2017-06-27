#include <gtest/gtest.h>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "TrainingSetTest.h"


using namespace Winzent::ANN;


TEST(TrainingSetTest, testOutputRelevant)
{
    ASSERT_TRUE(TrainingItem({ 1.0 }, { 1.0 }).outputRelevant());
    ASSERT_FALSE(TrainingItem(Vector()).outputRelevant());
}


TEST(TrainingSetTest, testJsonSerialization)
{
    TrainingSet ts;
    ts.targetError(0.01);
    ts.maxEpochs(1000);
    ts
            << TrainingItem({ 1.0, 1.0, 1.0 })
            << TrainingItem({ 2.0, 2.0, 2.0}, { 3.0, 3.0, 3.0 });

    auto json = to_json(ts);
    TrainingSet ts2{ from_json<TrainingSet>(json) };

    ASSERT_EQ(ts.error(), ts2.error());
    ASSERT_EQ(ts.maxEpochs(), ts2.maxEpochs());
    ASSERT_EQ(ts.trainingItems.size(), ts2.trainingItems.size());
}
