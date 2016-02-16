#include "Testrunner.h"

#include "Neuron.h"
#include "LinearActivationFunction.h"

#include "Layer.h"
#include "LayerTest.h"


using Winzent::ANN::Layer;
using Winzent::ANN::Neuron;
using Winzent::ANN::LinearActivationFunction;


LayerTest::LayerTest(QObject *parent) :
    QObject(parent)
{
}


void LayerTest::testLayerCreation()
{
    Layer layer;
    QCOMPARE(layer.size(), 0ul);
}


void LayerTest::testNeuronAddition()
{
    Layer layer;
    layer << new Neuron(new LinearActivationFunction());
    layer << new Neuron(new LinearActivationFunction());

    QCOMPARE(layer.size(), 2ul);
    QVERIFY(layer.neuronAt(0)->parent() == &layer);
    QVERIFY(layer.neuronAt(1)->parent() == &layer);

    for (const Neuron &n: layer) {
        QVERIFY(layer.contains(n));
    }
}


void LayerTest::testNeuronIterator()
{
    QList<const Neuron*> neurons;

    Layer layer;
    layer << new Neuron(new LinearActivationFunction());
    layer << new Neuron(new LinearActivationFunction());

    for (const Neuron &neuron: layer) {
        neurons << &neuron;
    }

    QCOMPARE(static_cast<size_t>(neurons.size()), layer.size());
}


TESTCASE(LayerTest)
