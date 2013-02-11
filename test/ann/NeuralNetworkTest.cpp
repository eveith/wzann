/*!
 * \file	NeuralNetworkTest.cpp
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#include <QtTest>
#include <QObject>

#define private public
#include "SigmoidActivationFunction.h"
#include "Neuron.h"
#include "Layer.h"
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
                            network->translateIndex(i-1, j+1),
                            network->translateIndex(i, k+1));
                    network->weight(
                            network->translateIndex(i-1, j+1),
                            network->translateIndex(i, k+1),
                            1.0);
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
    int numNeuronsPerLayer = 3;

    NeuralNetwork* network = new NeuralNetwork(this);

    Layer* l1 = new Layer(this);

    for (int i = 0; i != numNeuronsPerLayer; ++i) {
        l1->neurons << new Neuron(new SigmoidActivationFunction);
    }

    *network << l1;

    QVERIFY(network->inputLayer() == l1);
    QVERIFY(network->outputLayer() == l1);

    QVERIFY(l1->parent() == network);

    Layer* l2 = new Layer(this);

    for (int i = 0; i != numNeuronsPerLayer; ++i) {
        l2->neurons << new Neuron(new SigmoidActivationFunction);
    }

    *network << l2;

    QVERIFY(network->inputLayer() == l1);
    QVERIFY(network->outputLayer() == l2);

    QCOMPARE(network->m_weightMatrix.size(), l1->size() + l2->size());
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
