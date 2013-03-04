/*!
 * \file	NeuralNetworkTest.cpp
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#include "Testrunner.h"

#define private public
#include "SigmoidActivationFunction.h"
#include "Neuron.h"
#include "Layer.h"
#include "Connection.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"

#include "NeuralNetworkTest.h"


using namespace Winzent::ANN;



namespace Mock {
    NeuralNetworkTestDummyPattern::NeuralNetworkTestDummyPattern():
            NeuralNetworkPattern(QList<int>(), QList<ActivationFunction*>()),
            numLayers(3), numNeuronsPerLayer(13)
    {
        for (int i = 0; i != numLayers; ++i) {
            m_layerSizes << numNeuronsPerLayer;
            m_activationFunctions << new SigmoidActivationFunction();
        }
    }


    void NeuralNetworkTestDummyPattern::configureNetwork(
            NeuralNetwork* network)
    {
        // Connect half of the neurons of the nth layer with all neurons
        // of the (n+1)th layer.

        for (int i = 0; i != numLayers; ++i) {
            Layer* l = new Layer();

            for (int j = 0; j != numNeuronsPerLayer; ++j) {
                *l << new Neuron(new SigmoidActivationFunction());
            }

            *network << l;

            if (i == 0) {
                continue;
            }

            for (int j = 0; j != numNeuronsPerLayer/2; ++j) {
                for (int k = 0; k != numNeuronsPerLayer; ++k) {
                    network->connectNeurons(
                            network->layerAt(i-1)->neuronAt(j),
                            network->layerAt(i)->neuronAt(k))
                    ->weight(1.0);

                    QVERIFY(true == network->neuronConnectionExists(
                            network->layerAt(i-1)->neuronAt(j),
                            network->layerAt(i)->neuronAt(k)));
                    QCOMPARE(network->neuronConnection(
                                network->layerAt(i-1)->neuronAt(j),
                                network->layerAt(i)->neuronAt(k))->parent(),
                            network);
                }
            }
        }
    }


    ValueVector NeuralNetworkTestDummyPattern::calculate(
            NeuralNetwork*,
            const ValueVector& input)
    {
        return input;
    }


    NeuralNetworkPattern* NeuralNetworkTestDummyPattern::clone() const
    {
        return new NeuralNetworkTestDummyPattern();
    }
}


void NeuralNetworkTest::testLayerAdditionRemoval()
{
    NeuralNetwork* network = new NeuralNetwork(this);

    Layer* l1 = new Layer(this);
    *network << l1;

    QVERIFY(network->inputLayer() == l1);
    QVERIFY(network->outputLayer() == l1);

    QVERIFY(l1->parent() == network);

    Layer* l2 = new Layer(this);

    *network << l2;

    QVERIFY(network->inputLayer() == l1);
    QVERIFY(network->outputLayer() == l2);
    QVERIFY(2 == network->size());

    delete network;
}


void NeuralNetworkTest::testCalculateLayerTransition()
{
    NeuralNetwork* network = new NeuralNetwork();
    Mock::NeuralNetworkTestDummyPattern* pattern =
            new Mock::NeuralNetworkTestDummyPattern();

    network->configure(pattern);

    ValueVector inVector(pattern->numNeuronsPerLayer, 1.0);
    ValueVector outVector = network->calculateLayerTransition(0, 1, inVector);

    QCOMPARE(outVector.size(), pattern->numNeuronsPerLayer);

    for (int i = 0; i != outVector.size(); ++i) {
        QCOMPARE(static_cast<int>(outVector[i]),
                pattern->numNeuronsPerLayer/2);
    }

    delete network;
}


void NeuralNetworkTest::testCalculateLayer()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(&pattern);

    ValueVector inVector(pattern.numNeuronsPerLayer, 1.0);
    ValueVector outVector = network.calculateLayer(1, inVector);

    QCOMPARE(outVector.size(), pattern.numNeuronsPerLayer);
    QCOMPARE(outVector.size(), inVector.size());

    foreach (double d, outVector) {
        QCOMPARE(1.0 + d, 1.0 + SigmoidActivationFunction().calculate(2.0));
    }
}


void NeuralNetworkTest::testSerialization()
{
    QFile testResultFile(QString(QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);

    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(&pattern);

    testResultStream << network;
    testResultStream.flush();
    testResultFile.close();
}


void NeuralNetworkTest::testInitialLayerSize()
{
    Layer l;
    QCOMPARE(l.neurons.size(), 1);
    QCOMPARE(l.neurons.first()->activate(4123), 1.0);
}


void NeuralNetworkTest::testConnectionsFromTo()
{
    NeuralNetwork *network = new NeuralNetwork(this);

    Neuron *s = new Neuron(new SigmoidActivationFunction(), network);
    Neuron *d = new Neuron(new SigmoidActivationFunction(), network);

    Layer *l1 = new Layer(network);
    Layer *l2 = new Layer(network);

    *l1 << s;
    *l2 << d;

    *network << l1 << l2;

    network->connectNeurons(s, d);

    QCOMPARE(network->layerAt(0)->neuronAt(0), s);
    QCOMPARE(network->layerAt(1)->neuronAt(0), d);

    QCOMPARE(network->neuronConnectionsFrom(s).size(), 1);
    QCOMPARE(network->neuronConnectionsTo(d).size(), 2);

    QCOMPARE(network->neuronConnectionsFrom(s)[0]->destination(), d);
    QCOMPARE(network->neuronConnectionsTo(d)[1]->source(), s);
}


TESTCASE(NeuralNetworkTest)
