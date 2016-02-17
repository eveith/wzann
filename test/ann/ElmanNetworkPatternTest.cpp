#include "Testrunner.h"

#include <QTextStream>

#include <initializer_list>

#include "ElmanNetworkPatternTest.h"

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
    QObject(parent)
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
    NeuralNetwork network;
    ElmanNetworkPattern pattern(layers, activationFunctions);

    network.configure(pattern);

    QFile testResultFile(QString(
            QTest::currentTestFunction()).append(".out"));
    testResultFile.open(QIODevice::Text
            | QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream testResultStream(&testResultFile);
    testResultStream << network;
    testResultStream.flush();
    testResultFile.close();

    QCOMPARE(network.size(), 4ul);

    // Each hidden layer neuron has one context neuron: Check

    for (int i = 1; i != layers.at(1); ++i) {
        for (int j = 1; j != layers.at(1); ++j) {
            bool connection = network.connectionExists(
                    network[ElmanNetworkPattern::HIDDEN][i],
                    network[ElmanNetworkPattern::CONTEXT][j]);

            if (i == j) {
                QCOMPARE(connection, true);
                Connection *c = network.connection(
                        network[ElmanNetworkPattern::HIDDEN][i],
                        network[ElmanNetworkPattern::CONTEXT][j]);
                QCOMPARE(c->weight(), 1.0);
                QCOMPARE(c->fixedWeight(), true);
            } else {
                QCOMPARE(connection, false);
            }
        }
    }
}


TESTCASE(ElmanNetworkPatternTest)
