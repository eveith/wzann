/*!
 * \file	NeuralNetworkTest.cpp
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#include <QtDebug>

#include <ClassRegistry.h>

#include "Testrunner.h"

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
            NeuralNetwork &network)
    {
        // Connect all neurons of the nth layer with all neurons
        // of the (n+1)th layer.

        for (int i = 0; i != numLayers; ++i) {
            Layer* l = new Layer();

            for (int j = 0; j != numNeuronsInLayer(i); ++j) {
                Neuron *n = new Neuron(new LinearActivationFunction(1.0));
                *l << n;
            }

            network << l;

            if (0 == i) {
                continue;
            }

            for (int j = 0; j != numNeuronsInLayer(i-1); ++j) {
                for (int k = 0; k != numNeuronsInLayer(i); ++k) {
                    network.connectNeurons(
                            network[i-1][j],
                            network[i][k])
                        .weight(1.0);

                    QVERIFY(true == network.connectionExists(
                            network[i-1][j],
                            network[i][k]));
                }
            }
        }
    }


    Vector NeuralNetworkTestDummyPattern::calculate(
            NeuralNetwork &,
            const Vector &input)
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
    NeuralNetwork network;

    Layer* l1 = new Layer();
    network << l1;

    QVERIFY(&(network.inputLayer()) == l1);
    QVERIFY(&(network.outputLayer()) == l1);

    QVERIFY(l1->parent() == &network);

    Layer* l2 = new Layer();

    network << l2;

    QVERIFY(&(network.inputLayer()) == l1);
    QVERIFY(&(network.outputLayer()) == l2);
    QVERIFY(2ul == network.size());
}


void NeuralNetworkTest::testCalculateLayerTransition()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(pattern);

    const int fromLayer = 0;
    const int toLayer   = 1;

    Vector inVector(pattern.numNeuronsInLayer(fromLayer), 1.0);
    Vector outVector = network.calculateLayerTransition(
            network[fromLayer],
            network[toLayer],
            inVector);

    QCOMPARE(outVector.size(), pattern.numNeuronsInLayer(toLayer));

    for (int i = 0; i != outVector.size(); ++i) {
        QCOMPARE(outVector[i] + 1.0, 2.0); // BIAS neuron not added in here.
    }
}


void NeuralNetworkTest::testCalculateLayer()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(pattern);

    const qreal inValue = 1.0;
    const int layer     = 2;

    Vector inVector(pattern.numNeuronsInLayer(layer), inValue);
    Vector outVector = network[layer].activate(inVector);

    QCOMPARE(outVector.size(), pattern.numNeuronsInLayer(layer));
    QCOMPARE(outVector.size(), inVector.size());

    for (const qreal &d: outVector) {
        QCOMPARE(
                1.0 + d,
                1.0 + LinearActivationFunction().calculate(inValue));
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
    network.configure(pattern);

    testResultStream << network;
    testResultStream.flush();
    testResultFile.close();
}


void NeuralNetworkTest::testInitialLayerSize()
{
    Layer l;
    QCOMPARE(l.size(), 0ul);
}


void NeuralNetworkTest::testConnectionsFromTo()
{
    NeuralNetwork *network = new NeuralNetwork();

    Neuron *s = new Neuron(new LinearActivationFunction());
    Neuron *d = new Neuron(new LinearActivationFunction());

    Layer *l1 = new Layer();
    Layer *l2 = new Layer();

    *l1 << s;
    *l2 << d;

    *network << l1 << l2;

    network->connectNeurons(*s, *d);

    QCOMPARE(network->layerAt(0)->neuronAt(0), s);
    QCOMPARE(network->layerAt(1)->neuronAt(0), d);

    auto connectionSources = network->connectionsFrom(*s);
    auto connectionDestinations = network->connectionsTo(*d);
    QCOMPARE(connectionSources.second-connectionSources.first, 1l);
    QCOMPARE(connectionDestinations.second-connectionDestinations.first, 1l);

    QCOMPARE((*connectionSources.first)->destination(), *d);
    QCOMPARE((*connectionDestinations.first)->source(), *s);

    delete network;
}


void NeuralNetworkTest::testClone()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(pattern);

    NeuralNetwork *clone = network.clone();

    QCOMPARE(network.size(), clone->size());
    QVERIFY(&(network.biasNeuron()) != &(clone->biasNeuron()));

    for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
        Layer *origLayer    = network.layerAt(i);
        Layer *cloneLayer   = clone->layerAt(i);

        QVERIFY(origLayer != cloneLayer);
        QCOMPARE(origLayer->size(), cloneLayer->size());

        QCOMPARE(origLayer->parent(), &network);
        QCOMPARE(cloneLayer->parent(), clone);

        for (Layer::size_type j = 0; j < origLayer->size(); ++j) {
            Neuron *origNeuron  = origLayer->neuronAt(j);
            Neuron *cloneNeuron = cloneLayer->neuronAt(j);

            QCOMPARE(origNeuron->parent(), origLayer);
            QCOMPARE(cloneNeuron->parent(), cloneLayer);

            QVERIFY(origLayer->contains(*origNeuron));
            QVERIFY(cloneLayer->contains(*cloneNeuron));

            QVERIFY(network.contains(*origNeuron));
            QVERIFY(clone->contains(*cloneNeuron));

            QVERIFY(origNeuron->activationFunction()
                    != cloneNeuron->activationFunction());

            auto origConnections = network.connectionsFrom(*origNeuron);
            auto cloneConnections = clone->connectionsFrom(*cloneNeuron);

            QCOMPARE(
                    cloneConnections.second - cloneConnections.first,
                    origConnections.second - origConnections.first);
        }
    }

    delete clone;
}


void NeuralNetworkTest::testLayerIterator()
{
    QList<const Layer *> layers;

    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(pattern);

    for (auto &layer: boost::make_iterator_range(network.layers())) {
        layers.push_back(&layer);
    }

    QVERIFY(static_cast<size_t>(layers.size()) == network.size());
    QVERIFY(&(network.inputLayer()) == layers.first());
    QVERIFY(&(network.outputLayer()) == layers.last());

    layers.clear();

    for (auto &layer: boost::make_iterator_range(network.layers())) {
        if (&(network.inputLayer()) == &layer) {
            layers << &layer;
        }
    }

    QCOMPARE(layers.size(), 1);
    QVERIFY(&(network.inputLayer()) == layers.first());
}


void NeuralNetworkTest::testEachConnectionIterator()
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(pattern);
    QList<Connection *> connections;

    for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
        Layer &layer = network[i];

        for (Layer::size_type j = 0; j != layer.size(); ++j) {
            Neuron &n = layer[j];
            for (Connection *c: boost::make_iterator_range(
                     network.connectionsFrom(n))) {
                connections.append(c);
            }
        }
    }

    for (Connection *c: boost::make_iterator_range(
             network.connectionsFrom(network.biasNeuron()))) {
        connections.append(c);
    }

    int iterated = 0;
    network.eachConnection([&iterated, &connections](Connection *const &c) {
        iterated++;
        QVERIFY(connections.contains(c));
    });

    QCOMPARE(iterated, connections.size());
}


void NeuralNetworkTest::testOperatorEquals()
{
    NeuralNetwork n1, n2;
    QVERIFY(n1 == n2);
    QVERIFY(! (n1 != n2));

    Mock::NeuralNetworkTestDummyPattern pattern;
    n1.configure(pattern);
    QVERIFY(n1 != n2);
    n2.configure(pattern);
    QVERIFY(n1 == n2);

    n1.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(42.23);
        }
    });

    n2.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(1.0);
        }
    });

    QVERIFY(n1 != n2);

    n2.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(42.23);
        }
    });

    QVERIFY(n1 == n2);
}


void NeuralNetworkTest::testJsonSerialization()
{
    NeuralNetwork n1, n2;
    Mock::NeuralNetworkTestDummyPattern pattern;

    n1.configure(pattern);
    n1.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(12.20);
        }
    });

    QVERIFY(n1 != n2);
    n2.fromJSON(n1.toJSON());
    QVERIFY(n2 == n1);
}


TESTCASE(NeuralNetworkTest)
WINZENT_REGISTER_CLASS(
        Mock::NeuralNetworkTestDummyPattern,
        Winzent::ANN::NeuralNetworkPattern)
