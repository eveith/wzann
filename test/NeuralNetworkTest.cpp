#include <gtest/gtest.h>

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

                    ASSERT_TRUE(true == network.connectionExists(
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


TEST(NeuralNetworkTest, testLayerAdditionRemoval)
{
    NeuralNetwork network;

    Layer* l1 = new Layer();
    network << l1;

    ASSERT_TRUE(&(network.inputLayer()) == l1);
    ASSERT_TRUE(&(network.outputLayer()) == l1);

    ASSERT_TRUE(l1->parent() == &network);

    Layer* l2 = new Layer();

    network << l2;

    ASSERT_TRUE(&(network.inputLayer()) == l1);
    ASSERT_TRUE(&(network.outputLayer()) == l2);
    ASSERT_TRUE(2ul == network.size());
}


TEST(NeuralNetworkTest, testCalculateLayerTransition)
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

    ASSERT_EQ(pattern.numNeuronsInLayer(toLayer, outVector.size()));

    for (int i = 0; i != outVector.size(); ++i) {
        ASSERT_EQ(2.0, outVector[i] + 1.0); // BIAS neuron not added in here.
    }
}


TEST(NeuralNetworkTest, testCalculateLayer)
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(pattern);

    const qreal inValue = 1.0;
    const int layer     = 2;

    Vector inVector(pattern.numNeuronsInLayer(layer), inValue);
    Vector outVector = network[layer].activate(inVector);

    ASSERT_EQ(pattern.numNeuronsInLayer(layer, outVector.size()));
    ASSERT_EQ(inVector.size(, outVector.size()));

    for (const auto& d: outVector) {
        ASSERT_EQ(
                1.0 + LinearActivationFunction().calculate(inValue),
                1.0 + d);
    }
}


TEST(NeuralNetworkTest, testSerialization)
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


TEST(NeuralNetworkTest, testInitialLayerSize)
{
    Layer l;
    ASSERT_EQ(0ul, l.size());
}


TEST(NeuralNetworkTest, testConnectionsFromTo)
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

    ASSERT_EQ(s, network->layerAt(0)->neuronAt(0));
    ASSERT_EQ(d, network->layerAt(1)->neuronAt(0));

    auto connectionSources = network->connectionsFrom(*s);
    auto connectionDestinations = network->connectionsTo(*d);
    ASSERT_EQ(1l, connectionSources.second-connectionSources.first);
    ASSERT_EQ(1l, connectionDestinations.second-connectionDestinations.first);

    ASSERT_EQ(*d, (*connectionSources.first)->destination());
    ASSERT_EQ(*s, (*connectionDestinations.first)->source());

    delete network;
}


TEST(NeuralNetworkTest, testClone)
{
    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;

    network.configure(pattern);

    NeuralNetwork *clone = network.clone();

    ASSERT_EQ(clone->size(, network.size()));
    ASSERT_TRUE(&(network.biasNeuron()) != &(clone->biasNeuron()));

    for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
        Layer *origLayer    = network.layerAt(i);
        Layer *cloneLayer   = clone->layerAt(i);

        ASSERT_TRUE(origLayer != cloneLayer);
        ASSERT_EQ(cloneLayer->size(, origLayer->size()));

        ASSERT_EQ(&network, origLayer->parent());
        ASSERT_EQ(clone, cloneLayer->parent());

        for (Layer::size_type j = 0; j < origLayer->size(); ++j) {
            Neuron *origNeuron  = origLayer->neuronAt(j);
            Neuron *cloneNeuron = cloneLayer->neuronAt(j);

            ASSERT_EQ(origLayer, origNeuron->parent());
            ASSERT_EQ(cloneLayer, cloneNeuron->parent());

            ASSERT_TRUE(origLayer->contains(*origNeuron));
            ASSERT_TRUE(cloneLayer->contains(*cloneNeuron));

            ASSERT_TRUE(network.contains(*origNeuron));
            ASSERT_TRUE(clone->contains(*cloneNeuron));

            ASSERT_TRUE(origNeuron->activationFunction()
                    != cloneNeuron->activationFunction());

            auto origConnections = network.connectionsFrom(*origNeuron);
            auto cloneConnections = clone->connectionsFrom(*cloneNeuron);

            ASSERT_EQ(
                    origConnections.second - origConnections.first,
                    cloneConnections.second - cloneConnections.first);
        }
    }

    delete clone;
}


TEST(NeuralNetworkTest, testLayerIterator)
{
    QList<const Layer *> layers;

    NeuralNetwork network;
    Mock::NeuralNetworkTestDummyPattern pattern;
    network.configure(pattern);

    for (auto &layer: boost::make_iterator_range(network.layers())) {
        layers.push_back(&layer);
    }

    ASSERT_TRUE(static_cast<size_t>(layers.size()) == network.size());
    ASSERT_TRUE(&(network.inputLayer()) == layers.first());
    ASSERT_TRUE(&(network.outputLayer()) == layers.last());

    layers.clear();

    for (auto &layer: boost::make_iterator_range(network.layers())) {
        if (&(network.inputLayer()) == &layer) {
            layers << &layer;
        }
    }

    ASSERT_EQ(1, layers.size());
    ASSERT_TRUE(&(network.inputLayer()) == layers.first());
}


TEST(NeuralNetworkTest, testEachConnectionIterator)
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
        ASSERT_TRUE(connections.contains(c));
    });

    ASSERT_EQ(connections.size(, iterated));
}


TEST(NeuralNetworkTest, testOperatorEquals)
{
    NeuralNetwork n1, n2;
    ASSERT_TRUE(n1 == n2);
    ASSERT_FALSE((n1 != n2));

    Mock::NeuralNetworkTestDummyPattern pattern;
    n1.configure(pattern);
    ASSERT_TRUE(n1 != n2);
    n2.configure(pattern);
    ASSERT_TRUE(n1 == n2);

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

    ASSERT_TRUE(n1 != n2);

    n2.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(42.23);
        }
    });

    ASSERT_TRUE(n1 == n2);
}


TEST(NeuralNetworkTest, testJsonSerialization)
{
    NeuralNetwork n1, n2;
    Mock::NeuralNetworkTestDummyPattern pattern;

    n1.configure(pattern);
    n1.eachConnection([](Connection* const& c) {
        if (! c->fixedWeight()) {
            c->weight(12.20);
        }
    });

    ASSERT_TRUE(n1 != n2);
    n2.fromJSON(n1.toJSON());
    ASSERT_TRUE(n2 == n1);
}
