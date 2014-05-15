#include "Testrunner.h"

#include "Neuron.h"
#include "LayerTest.h"

#define private public
#include "Layer.h"


using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;


LayerTest::LayerTest(QObject *parent) :
    QObject(parent)
{
}


void LayerTest::testLayerCreation()
{
    Layer layer;
    QCOMPARE(layer.size(), 0);
}


void LayerTest::testNeuronAddition()
{
    Layer layer;
    layer << new Neuron(NULL, this);
    layer << new Neuron(NULL, this);

    QCOMPARE(layer.size(), 2);
    QVERIFY(layer.neuronAt(0)->parent() == &layer);
}


void LayerTest::testNeuronIterator()
{
    QList<Neuron *> neurons;

    Layer layer;
    layer << new Neuron(NULL, this);
    layer << new Neuron(NULL, this);

    layer.eachNeuron([&](Neuron *const &neuron) {
        neurons << neuron;
    });

    QCOMPARE(neurons.size(), layer.size());
}


TESTCASE(LayerTest)
