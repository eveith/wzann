#include <QObject>
#include <QtTest>
#include <QDebug>

#include "NeuralNetwork.h"
#include "Layer.h"
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
    NeuralNetwork *network = new NeuralNetwork(this);
    PerceptronNetworkPattern *pattern = new PerceptronNetworkPattern({
                2,
                3,
                1
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            },
            network);
    network->configure(pattern);

    for (int i = 0; i != network->size() - 1; ++i) {
        for (int j = 0; j != (*network)[i]->size(); ++j) {
            for (int k = 0; k != (*network)[i+1]->size(); ++k) {
                QVERIFY(network->neuronConnectionExists(
                    network->translateIndex(i, j),
                    network->translateIndex(i+1, k)));
            }
        }
    }

    QVERIFY(! network->neuronConnectionExists(
            network->translateIndex(1, 0),
            network->translateIndex(0, 0)));
    QVERIFY(! network->neuronConnectionExists(
            network->translateIndex(1, 0),
            network->translateIndex(1, 0)));
    QVERIFY(! network->neuronConnectionExists(
            network->translateIndex(1, 0),
            network->translateIndex(1, 1)));
}


void PerceptronNetworkPatternTest::testCalculate()
{
    NeuralNetwork *network = new NeuralNetwork(this);
    PerceptronNetworkPattern *pattern = new PerceptronNetworkPattern({
                2,
                3,
                1
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            },
            network);
    network->configure(pattern);

    ValueVector input = { 1.0, 0.0 };
    ValueVector output = network->calculate(input);
    qDebug() << output[0];
}
