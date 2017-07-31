#include <gtest/gtest.h>

#include <string>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "TestSchemaPath.h"
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
    TrainingSet ts2{ from_json<TrainingSet>(
            json,
            WZANN_SCHEMA_PATH "/TrainingSetSchema.json") };

    ASSERT_EQ(ts.error(), ts2.error());
    ASSERT_EQ(ts.maxEpochs(), ts2.maxEpochs());
    ASSERT_EQ(ts.trainingItems.size(), ts2.trainingItems.size());
}


TEST(TrainingSetTest, testJsonSchema)
{
    static std::string wrongTsJson{ "{ \"targetError\": 0.01 }" };
    bool hasThrown = false;

    try {
        TrainingSet ts{ from_json<TrainingSet>(
                wrongTsJson,
                WZANN_SCHEMA_PATH "/TrainingSetSchema.json") };
    } catch (SchemaValidationException const&) {
        hasThrown = true;
    }

    ASSERT_TRUE(hasThrown);
}
