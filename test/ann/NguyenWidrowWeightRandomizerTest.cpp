#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "SigmoidActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "NguyenWidrowWeightRandomizer.h"

#include "Testrunner.h"

#include "NguyenWidrowWeightRandomizerTest.h"


using namespace Winzent::ANN;


NguyenWidrowWeightRandomizerTest::NguyenWidrowWeightRandomizerTest(
        QObject *parent):
            QObject(parent)
{
}



void NguyenWidrowWeightRandomizerTest::testRandomizeWeights()
{
    NeuralNetwork *network = new NeuralNetwork();
    PerceptronNetworkPattern *pattern = new PerceptronNetworkPattern({
            1,
            2,
            3
            }, {
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction(),
                new SigmoidActivationFunction()
            });
    network->configure(pattern);

    NguyenWidrowWeightRandomizer().randomize(*network);

    for (int i = 0; i != network->size(); ++i) {
        Layer *layer = network->layerAt(i);

        for (size_t j = 0; j != layer->size(); ++j) {
            Neuron *neuron = layer->neuronAt(j);
            for (const auto &c: network->neuronConnectionsFrom(neuron)) {
                QVERIFY(1.0 != 1.0 + c->weight());
            }
        }

        for (const auto &c: network->neuronConnectionsFrom(
                network->biasNeuron())) {
            QCOMPARE(c->weight(), -1.0);
        }
    }

    delete network;
    delete pattern;
}


TESTCASE(NguyenWidrowWeightRandomizerTest);
