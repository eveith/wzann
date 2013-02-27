#include "NeuralNetwork.h"
#include "PerceptronNetworkPattern.h"
#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"
#include "TrainingSet.h"
#include "BackpropagationTrainingAlgorithm.h"

#include "Testrunner.h"
#include "BackpropagationTrainingAlgorithmTest.h"


using namespace Winzent::ANN;


BackpropagationTrainingAlgorithmTest::BackpropagationTrainingAlgorithmTest(
        QObject *parent):
            QObject(parent)
{
}


void BackpropagationTrainingAlgorithmTest::testTrainXOR()
{
    NeuralNetwork *network = new NeuralNetwork(this);
    PerceptronNetworkPattern *pattern = new PerceptronNetworkPattern(
            {
                2,
                3,
                1
            }, {
                new LinearActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            },
            this);

    network->configure(pattern);

    // Build training data:

    QList<TrainingItem> trainingItems;
    trainingItems
            << TrainingItem(ValueVector({ 0.0, 0.0 }), ValueVector({ 0.0 }))
            << TrainingItem(ValueVector({ 0.0, 1.0 }), ValueVector({ 1.0 }))
            << TrainingItem(ValueVector({ 1.0, 1.0 }), ValueVector({ 0.0 }))
            << TrainingItem(ValueVector({ 1.0, 0.0 }), ValueVector({ 1.0 }));

    TrainingSet *trainingSet = new TrainingSet(
            trainingItems,
            0.7,
            0.001,
            50000);

    network->train(new BackpropagationTrainingAlgorithm(this), trainingSet);
}


TESTCASE(BackpropagationTrainingAlgorithmTest);
