#include <QtTest>

#include "Layer.h"
#include "Neuron.h"
#include "LayerTest.h"


using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;


LayerTest::LayerTest(QObject *parent) :
    QObject(parent)
{
}


void LayerTest::testLayerCreation()
{
    Layer layer;
    QVERIFY2(layer.neurons.size() == 1,
            "A layer shall include a bias neuron by default");
    QCOMPARE(layer.biasNeuron(), layer.neurons.last());
}


void LayerTest::testNeuronAddition()
{
    Layer layer;
    layer << new Neuron(NULL, this);
    layer << new Neuron(NULL, this);

    QVERIFY2(layer.size() == layer.neurons.size() - 1,
        "Layer::size() should exclude the bias neuron");
    QVERIFY2(layer.biasNeuron() == layer.neurons.last(),
        "The bias neuron shall always be at the last index");
}
