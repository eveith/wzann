#include <QtTest>
#include <QObject>

#include "Testrunner.h"

#include "Connection.h"
#include "NeuralNetwork.h"
#include "LinearActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "SimpleWeightRandomizer.h"
#include "SimpleWeightRandomizerTest.h"


using Winzent::ANN::Connection;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::SimpleWeightRandomizer;
using Winzent::ANN::LinearActivationFunction;
using Winzent::ANN::PerceptronNetworkPattern;


SimpleWeightRandomizerTest::SimpleWeightRandomizerTest(QObject *parent):
        QObject(parent)
{
}


void SimpleWeightRandomizerTest::testWeightRandomization()
{
    NeuralNetwork neuralNetwork;
    PerceptronNetworkPattern pattern({
                5,
                20
            }, {
                new LinearActivationFunction(),
                new LinearActivationFunction()
            });
    neuralNetwork.configure(pattern);

    neuralNetwork.eachConnection(
            [&neuralNetwork](const Connection *const &c) {
        if (c->source() != neuralNetwork.biasNeuron()) {
            QCOMPARE(1.0, c->weight() + 1.0);
        }
    });

    SimpleWeightRandomizer swr;
    swr.randomize(neuralNetwork);

    neuralNetwork.eachConnection(
                [&neuralNetwork, &swr](const Connection *const &c) {
        if (c->source() != neuralNetwork.biasNeuron()) {
            QVERIFY(1.0 != c->weight() + 1.0);
            QVERIFY(swr.minWeight() <= c->weight());
            QVERIFY(c->weight() <= swr.maxWeight());
        }
    });
}


TESTCASE(SimpleWeightRandomizerTest)
