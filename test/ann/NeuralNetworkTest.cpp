/*!
 * \file	NeuralNetworkTest.cpp
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#include <QtDebug>

#include "Testrunner.h"

#define private public
#include "LinearActivationFunction.h"
#include "Neuron.h"
#include "Layer.h"
#include "Connection.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"

#include "NeuralNetworkTest.h"


using namespace Winzent::ANN;


const int Mock::NeuralNetworkTestDummyPattern::numLayers = 3;


namespace Mock {
    NeuralNetworkTestDummyPattern::NeuralNetworkTestDummyPattern():
            NeuralNetworkPattern(QList<int>(), QList<ActivationFunction*>())
    {
        for (int i = 0; i != numLayers; ++i) {
            m_layerSizes << numNeuronsInLayer(i);
            m_activationFunctions << new LinearActivationFunction();
        }
    }


    void NeuralNetworkTestDummyPattern::configureNetwork(
            NeuralNetwork* network)
    {
        // Connect half of the neurons of the nth layer with all neurons
        // of the (n+1)th layer.

        for (int i = 0; i != numLayers; ++i) {
            Layer* l = new Layer();

            for (int j = 0; j != numNeuronsInLayer(i); ++j) {
                Neuron *n = new Neuron(
                        new LinearActivationFunction(1.0, this));
                n->cacheSize(1);
                *l << n;
            }

            *network << l;

            if (0 == i) {
                continue;
            }

            for (int j = 0; j != numNeuronsInLayer(i-1); ++j) {
                for (int k = 0; k != numNeuronsInLayer(i); ++k) {
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
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(&pattern);

    const int fromLayer = 0;
    const int toLayer   = 1;

    ValueVector inVector(pattern.numNeuronsInLayer(fromLayer), 1.0);
    ValueVector outVector = network.calculateLayerTransition(
            fromLayer,
            toLayer,
            inVector);

    QCOMPARE(outVector.size(), pattern.numNeuronsInLayer(toLayer));

    for (int i = 0; i != outVector.size(); ++i) {
        QCOMPARE(outVector[i], 1.0);
    }
}


void NeuralNetworkTest::testCalculateLayer()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(&pattern);

    const qreal inValue = 1.0;
    const int layer     = 2;

    ValueVector inVector(pattern.numNeuronsInLayer(layer), inValue);
    ValueVector outVector = network.calculateLayer(layer, inVector);

    QCOMPARE(outVector.size(), pattern.numNeuronsInLayer(layer));
    QCOMPARE(outVector.size(), inVector.size());

    foreach (double d, outVector) {
        QCOMPARE(
                1.0 + d,
                1.0 + LinearActivationFunction().calculate(inValue)
                    + -1.0 * LinearActivationFunction().calculate(1.0));
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

    Neuron *s = new Neuron(new LinearActivationFunction(), network);
    Neuron *d = new Neuron(new LinearActivationFunction(), network);

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


void NeuralNetworkTest::testClone()
{
    NeuralNetwork *network = new NeuralNetwork();
    Mock::NeuralNetworkTestDummyPattern *pattern =
            new Mock::NeuralNetworkTestDummyPattern();

    network->configure(pattern);

    NeuralNetwork *clone = network->clone();

    QCOMPARE(network->size(), clone->size());

    for (int i = 0; i != network->size(); ++i) {
        Layer *origLayer    = network->layerAt(i);
        Layer *cloneLayer   = clone->layerAt(i);

        QVERIFY(origLayer != cloneLayer);
        QCOMPARE(origLayer->size(), cloneLayer->size());

        QCOMPARE(origLayer->parent(), network);
        QCOMPARE(cloneLayer->parent(), clone);

        QVERIFY(NULL != cloneLayer->biasNeuron());
        QVERIFY(origLayer->biasNeuron() != cloneLayer->biasNeuron());

        QCOMPARE(origLayer->biasNeuron()->parent(), origLayer);
        QCOMPARE(cloneLayer->biasNeuron()->parent(), cloneLayer);

        for (int j = 0; j <= origLayer->size(); ++j) {
            Neuron *origNeuron  = origLayer->neuronAt(j);
            Neuron *cloneNeuron = cloneLayer->neuronAt(j);

            QCOMPARE(origNeuron->parent(), origLayer);
            QCOMPARE(cloneNeuron->parent(), cloneLayer);

            QVERIFY(origLayer->contains(origNeuron));
            QVERIFY(cloneLayer->contains(cloneNeuron));

            QVERIFY(network->containsNeuron(origNeuron));
            QVERIFY(clone->containsNeuron(cloneNeuron));

            QVERIFY(origNeuron->activationFunction()
                    != cloneNeuron->activationFunction());

            QList<Connection *> origConnections =
                    network->neuronConnectionsFrom(origNeuron);
            QList<Connection *> cloneConnections =
                    clone->neuronConnectionsFrom(cloneNeuron);

            QCOMPARE(cloneConnections.size(), origConnections.size());
        }
    }

    delete network;
    delete clone;
}


void NeuralNetworkTest::testEachConnectionIterator()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(&pattern);
    QList<Connection *> connections;

    for (int i = 0; i != network.size(); ++i) {
        Layer *l = network.layerAt(i);

        for (int j = 0; j != l->size() + 1; ++j) {
            Neuron *n = l->neuronAt(j);
            connections.append(network.neuronConnectionsFrom(n));
        }
    }

    int iterated = 0;
    network.eachConnection([&iterated, &connections](Connection *const &c) {
        iterated++;
        QVERIFY(connections.contains(c));
    });

    QCOMPARE(iterated, connections.size());
}


TESTCASE(NeuralNetworkTest)
