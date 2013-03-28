#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "SigmoidActivationFunction.h"
#include "PerceptronNetworkPattern.h"

#include "NguyenWidrowWeightRandomizer.h"

#include "Testrunner.h"

#include "NguyenWidrowWeightRandomizerTest.h"

#include <iostream>
using namespace std;

using namespace Winzent::ANN;


NguyenWidrowWeightRandomizerTest::NguyenWidrowWeightRandomizerTest(
        QObject *parent):
            QObject(parent)
{
}



void NguyenWidrowWeightRandomizerTest::testRandomizeWeights()
{
    qsrand(time(0));

    NeuralNetwork network;
    PerceptronNetworkPattern pattern({ 1, 2, 3 }, {
                new SigmoidActivationFunction(1.0, this),
                new SigmoidActivationFunction(1.0, this),
                new SigmoidActivationFunction(1.0, this)
            });
    network.configure(&pattern);

    NguyenWidrowWeightRandomizer().randomize(&network);

    for (int i = 0; i != network.size(); ++i) {
        Layer *layer = network.layerAt(i);

        for (int j = 0; j != layer->size(); ++j) {
            Neuron *neuron = layer->neuronAt(j);
            foreach (Connection *c,
                    network.neuronConnectionsFrom(neuron)) {
                QVERIFY(1.0 != 1.0 + c->weight());
                cout << c->weight() << endl;
            }
        }

        foreach (Connection *c,
                network.neuronConnectionsFrom(layer->biasNeuron())) {
            QCOMPARE(c->weight(), -1.0);
        }
    }
}


TESTCASE(NguyenWidrowWeightRandomizerTest);
