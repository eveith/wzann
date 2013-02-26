#include "Testrunner.h"

#include <QTextStream>

#include <initializer_list>

#include "ElmanNetworkPatternTest.h"

#define private public
#include "ElmanNetworkPattern.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Connection;
using Winzent::ANN::ElmanNetworkPattern;
using Winzent::ANN::ActivationFunction;
using Winzent::ANN::SigmoidActivationFunction;


ElmanNetworkPatternTest::ElmanNetworkPatternTest(QObject *parent) :
    QObject(parent),
    layers(QList<int>()),
    activationFunctions(QList<ActivationFunction*>())
{
    layers
            << 1
            << 10
            << 1;
    activationFunctions
            << new SigmoidActivationFunction()
            << new SigmoidActivationFunction()
            << new SigmoidActivationFunction();
}


void ElmanNetworkPatternTest::testConfigure()
{
    qsrand(time(NULL));
    NeuralNetwork network;
    ElmanNetworkPattern pattern(layers, activationFunctions);

    network.configure(&pattern);

    QFile testResultFile(QString(QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);
    testResultStream << network;
    testResultStream.flush();
    testResultFile.close();

    QCOMPARE(network.size(), 4);
    QCOMPARE(network.m_connectionSources.size(),
             layers[0] + 2* (layers[1]+1) + layers[2] +1);

    // Each input layer neuron has one context neuron: Check

    for (int i = 1; i != layers.at(1); ++i) {
        for (int j = 1; j != layers.at(1); ++j) {
            bool connection = network.neuronConnectionExists(
                   network.layerAt(ElmanNetworkPattern::HIDDEN)->neuronAt(i),
                   network.layerAt(ElmanNetworkPattern::CONTEXT)->neuronAt(j));

            if (i == j) {
                QCOMPARE(connection, true);
                Connection *c = network.neuronConnection(
                    network.layerAt(ElmanNetworkPattern::HIDDEN)->neuronAt(i),
                    network.layerAt(ElmanNetworkPattern::CONTEXT)->neuronAt(j));
                QCOMPARE(c->weight(), 1.0);
                QCOMPARE(c->fixedWeight(), true);
            } else {
                QCOMPARE(connection, false);
            }
        }
    }
}


TESTCASE(ElmanNetworkPatternTest)
