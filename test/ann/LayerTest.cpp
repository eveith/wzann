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
    QVERIFY2(layer.m_neurons.size() == 1,
            "A layer shall include a bias neuron by default");
    QCOMPARE(layer.biasNeuron(), layer.m_neurons.last());
}


void LayerTest::testNeuronAddition()
{
    Layer layer;
    layer << new Neuron(NULL, this);
    layer << new Neuron(NULL, this);

    QVERIFY2(layer.size() == layer.m_neurons.size() - 1,
        "Layer::size() should exclude the bias neuron");
    QVERIFY2(layer.biasNeuron() == layer.m_neurons.last(),
        "The bias neuron shall always be at the last index");
    QVERIFY(layer.m_neurons.at(0)->parent() == &layer);
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
    QVERIFY(!neurons.contains(layer.biasNeuron()));
}


TESTCASE(LayerTest)
