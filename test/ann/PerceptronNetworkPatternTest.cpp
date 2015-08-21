#include <QDebug>

#include "Testrunner.h"

#include "Layer.h"
#include "NeuralNetwork.h"
#include "SigmoidActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "PerceptronNetworkPatternTest.h"


using namespace Winzent::ANN;


PerceptronNetworkPatternTest::PerceptronNetworkPatternTest(QObject *parent) :
    QObject(parent)
{
}


void PerceptronNetworkPatternTest::testConfigure()
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern({
                2,
                3,
                1
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });
    network.configure(pattern);

    for (size_t i = 0; i != network.size() - 1; ++i) {
        for (size_t j = 0; j != network.layerAt(i)->size(); ++j) {
            for (size_t k = 0; k != network[i+1].size(); ++k) {
                QVERIFY(network.neuronConnectionExists(
                    network.layerAt(i)->neuronAt(j),
                    network.layerAt(i+1)->neuronAt(k)));
            }
        }
    }

    QVERIFY(! network.neuronConnectionExists(
            network.layerAt(1)->neuronAt(0),
            network.layerAt(0)->neuronAt(0)));
    QVERIFY(! network.neuronConnectionExists(
            network.layerAt(1)->neuronAt(0),
            network.layerAt(1)->neuronAt(0)));
    QVERIFY(! network.neuronConnectionExists(
            network.layerAt(1)->neuronAt(0),
            network.layerAt(1)->neuronAt(1)));
}


void PerceptronNetworkPatternTest::testCalculate()
{
    NeuralNetwork network;
    PerceptronNetworkPattern pattern({
                2,
                3,
                1
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });
    network.configure(pattern);

    Vector input = { 1.0, 0.0 };
    Vector output = network.calculate(input);
    QVERIFY(1.0f != output.first() + 1.0);
}


TESTCASE(PerceptronNetworkPatternTest)
