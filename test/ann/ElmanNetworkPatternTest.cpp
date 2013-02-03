#include <QtTest>
#include <QTextStream>

#include <initializer_list>

#include "ElmanNetworkPatternTest.h"

#define private public
#include "ElmanNetworkPattern.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Weight;
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

    QCOMPARE(network.m_layers.size(), 4);
    QCOMPARE(network.m_weightMatrix.size(),
             layers[0] + 2* layers[1] + layers[2]);

    // Each input layer neuron has one context neuron: Check

    for (int i = 0; i != layers.at(1); ++i) {
        for (int j = 0; j != layers.at(1); ++j) {
            bool connection = network.neuronConnectionExists(
                   network.translateIndex(ElmanNetworkPattern::HIDDEN, i),
                   network.translateIndex(ElmanNetworkPattern::CONTEXT, j));

            if (i == j) {
                QCOMPARE(connection, true);
                Weight *w = network.weight(
                    network.translateIndex(ElmanNetworkPattern::HIDDEN, i),
                    network.translateIndex(ElmanNetworkPattern::CONTEXT, j));
                QCOMPARE(w->weight(), 1.0);
                QCOMPARE(w->fixed, true);
            } else {
                QCOMPARE(connection, false);
            }
        }
    }
}
