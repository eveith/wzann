/*!
 * \file	NeuralNetwork.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include <QDebug>

#include <QObject>
#include <QList>
#include <QVector>
#include <QByteArray>
#include <QTextStream>

#include <qjson/serializer.h>
#include <qjson/qobjecthelper.h>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "Exception.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "TrainingAlgorithm.h"


namespace Winzent
{
    namespace ANN
    {

        const char NeuralNetwork::VERSION[] = "1.0";


        NeuralNetwork::NeuralNetwork(QObject* parent):
                QObject(parent),
                m_layers(QList<Layer*>()),
                m_connectionSources(QHash<Neuron*, QList<Connection*> >()),
                m_connectionDestinations(QHash<Neuron*, QList<Connection*> >()),
                m_pattern(NULL)
        {
            Q_ASSERT(m_connectionSources.size()
                     == m_connectionDestinations.size());
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork& rhs):
                QObject(rhs.parent()),
                m_layers(QList<Layer*>(rhs.m_layers)),
                m_connectionSources(QHash<Neuron*, QList<Connection*> >()),
                m_connectionDestinations(QHash<Neuron*, QList<Connection*> >()),
                m_pattern(NULL)
        {
            // Clone layers:

            foreach(Layer *l, rhs.m_layers) {
                Layer *layerClone = l->clone();
                layerClone->setParent(this);
                m_layers << layerClone;
            }

            // Clone connections. We have already cloned the neurons, which
            // means that we cannot simply clone all connections because we'd
            // end up with pointers to the original. Instead, we need to map
            // via the index in the layer and create connections of our own:

            foreach (Neuron *foreignNeuron, rhs.m_connectionSources.keys()) {

                // First, find the index of the source neuron:

                int srcLayerIndex = -1;
                int srcNeuronIndex = -1;

                for (int i = 0; i != rhs.size(); ++i) {
                    if (!rhs[i]->contains(foreignNeuron)) {
                        continue;
                    }

                    srcLayerIndex = i;
                    srcNeuronIndex = rhs[i]->neurons.indexOf(foreignNeuron);
                    break;
                }

                Q_ASSERT(srcLayerIndex >= 0 && srcNeuronIndex >= 0);

                // Then, find the index of the destination neurons:

                QList<Connection*> connections =
                        rhs.m_connectionDestinations[foreignNeuron];
                foreach (Connection *c, connections) {
                    int dstLayerIndex = -1;
                    int dstNeuronIndex = -1;

                    for (int i = 0; i != rhs.size(); ++i) {
                        if (!rhs[i]->contains(c->destination())) {
                            continue;
                        }

                        dstLayerIndex = i;
                        dstNeuronIndex = rhs[i]->neurons.indexOf(
                                c->destination());
                        break;
                    }

                    Q_ASSERT(dstLayerIndex >= 0 && dstNeuronIndex >= 0);

                    // Now, re-create the connection here:

                    Neuron *srcNeuron =
                            layerAt(srcLayerIndex)->neuronAt(srcNeuronIndex);
                    Neuron *dstNeuron =
                            layerAt(dstLayerIndex)->neuronAt(dstNeuronIndex);

                    Connection *newConnection = connectNeurons(
                            srcNeuron,
                            dstNeuron);
                    newConnection->weight(c->weight());
                    newConnection->fixedWeight(c->fixedWeight());
                    newConnection->setParent(this);
                }
            }

            // Make sure the cloned pattern has the correct parent object:

            if (NULL != rhs.m_pattern) {
                m_pattern = rhs.m_pattern->clone();
                m_pattern->setParent(this);
            }
        }


        NeuralNetwork::~NeuralNetwork()
        {
        }


        bool NeuralNetwork::containsNeuron(const Neuron *neuron) const
        {
            if (NULL == neuron) {
                return false;
            }

            foreach (Layer *l, m_layers) {
                if (l->contains(neuron)) {
                    return true;
                }
            }

            return false;
        }


        bool NeuralNetwork::neuronConnectionExists(
                const Neuron *from,
                const Neuron *to) const
        {
            if (! m_connectionSources.contains(const_cast<Neuron*>(from))) {
                return false;
            }

            QList<Connection*> connections =
                    m_connectionSources[const_cast<Neuron*>(from)];

            foreach (Connection *c, connections) {
                if (c->destination() == const_cast<Neuron*>(to)) {
                    return true;
                }
            }

            return false;
        }


        Connection* NeuralNetwork::neuronConnection(
                const Neuron *from,
                const Neuron *to)
                    const
                    throw(NoConnectionException)
        {
            if (!neuronConnectionExists(from, to)) {
                throw NoConnectionException();
            }

            Connection* result = NULL;

            QList<Connection*> connections =
                    m_connectionSources[const_cast<Neuron*>(from)];
            foreach(Connection *c, connections) {
                if (c->destination() == const_cast<Neuron*>(to)) {
                    result = c;
                }
            }

            return result;
        }


        void NeuralNetwork::weight(
                const Neuron *&from,
                const Neuron *&to,
                double value)
                    throw(NoConnectionException, WeightFixedException)
        {
            neuronConnection(from, to)->weight(value);
        }


        QList<Connection*> NeuralNetwork::neuronConnectionsFrom(
                const Neuron *neuron)
                    const
                    throw(UnknownNeuronException)
        {
            Q_ASSERT(NULL != neuron);

            if (!containsNeuron(neuron)) {
                throw UnknownNeuronException(neuron);
            }

            return QList<Connection*>(
                    m_connectionSources[const_cast<Neuron*>(neuron)]);
        }


        QList<Connection*> NeuralNetwork::neuronConnectionsTo(
                const Neuron *neuron)
                    const
                    throw(UnknownNeuronException)
        {
            Q_ASSERT(NULL != neuron);

            if (!containsNeuron(neuron)) {
                throw UnknownNeuronException(neuron);
            }

            return QList<Connection*>(
                    m_connectionDestinations[const_cast<Neuron*>(neuron)]);
        }


        Connection *NeuralNetwork::connectNeurons(Neuron *from, Neuron *to)
                throw(UnknownNeuronException)
        {
            if (!containsNeuron(from)) {
                throw UnknownNeuronException(from);
            }

            if (!containsNeuron(to)) {
                throw UnknownNeuronException(to);
            }

            Connection *connection = new Connection(from, to, 0.0, this);

            Q_ASSERT(connection->source() == from);
            Q_ASSERT(connection->destination() == to);

            if (!m_connectionSources.contains(from)) {
                m_connectionSources[from] = QList<Connection*>();
            }
            m_connectionSources[from] << connection;

            if (!m_connectionDestinations.contains(to)) {
                m_connectionDestinations[to] = QList<Connection*>();
            }
            m_connectionDestinations[to] << connection;

            return connection;
        }


        NeuralNetwork& NeuralNetwork::operator<<(Layer *layer)
        {
            m_layers << layer;
            layer->setParent(this);

            // Make sure the bias connection exists:

            Neuron *bias = layer->biasNeuron();

            for (int i = 0; i != layer->size(); ++i) {
                Connection *c = connectNeurons(bias, layer->neuronAt(i));
                c->weight(-1.0);
                c->fixedWeight(true);
            }

            return *this;
        }


        Layer*& NeuralNetwork::layerAt(const int &index)
        {
            return m_layers[index];
        }


        Layer* NeuralNetwork::layerAt(const int &index) const
        {
            return m_layers.at(index);
        }


        Layer*& NeuralNetwork::operator [](const int &index)
        {
            return layerAt(index);
        }


        Layer *NeuralNetwork::operator [](const int &index) const
        {
            return layerAt(index);
        }


        Layer* NeuralNetwork::inputLayer() const
        {
            return m_layers.first();
        }


        Layer* NeuralNetwork::outputLayer() const
        {
            return m_layers.last();
        }


        void NeuralNetwork::configure(const NeuralNetworkPattern *pattern)
        {
            // Get rid of the old pattern, if one exists:

            if (NULL != m_pattern) {
                delete m_pattern;
            }

            m_pattern = pattern->clone();
            m_pattern->setParent(this);

            m_pattern->configureNetwork(this);
        }


        void NeuralNetwork::train(
                TrainingAlgorithm *trainingStrategy,
                TrainingSet *trainingSet)
        {
            trainingStrategy->train(this, trainingSet);
        }


        ValueVector NeuralNetwork::calculateLayer(
                Layer *layer,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            if (layer->size() != input.size()) {
                throw LayerSizeMismatchException(input.size(), layer->size());
            }

            ValueVector output;
            output.fill(0.0, layer->size());

            Neuron *bias = layer->biasNeuron();

            for (int i = 0; i != input.size(); ++i) {
                double sum = input.at(i);
                Neuron *neuron = layer->neuronAt(i);

                // Add bias neuron. We ignore the bias neuron in the input layer
                // even when it's there; it does not make sense to include the
                // bias neuron in the input layer since its output would be
                // overwritten by the input anyways.

                if(m_layers.first() != layer) {
                    if (neuronConnectionExists(bias, neuron)) {
                        sum += neuronConnection(bias, neuron)->weight()
                                * bias->activate(1.0);
                    }
                }

                output[i] = neuron->activate(sum);
            }

            return output;
        }


        ValueVector NeuralNetwork::calculateLayer(
                const int &layerIndex,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            return this->calculateLayer(layerAt(layerIndex), input);
        }


        ValueVector NeuralNetwork::calculateLayerTransition(
                const int &from,
                const int &to,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            Layer *fromLayer    = layerAt(from);
            Layer *toLayer      = layerAt(to);
            int fromLayerSize   = fromLayer->size();
            int toLayerSize     = toLayer->size();

            if (input.size() != fromLayerSize) {
                throw LayerSizeMismatchException(input.size(), fromLayerSize);
            }

            ValueVector output;
            output.fill(0.0, toLayerSize);
            Q_ASSERT(output.size() == toLayerSize);

            for (int i = 0; i != fromLayerSize; ++i) {
                Neuron *fromNeuron = fromLayer->neuronAt(i);
                QList<Connection*> connections =
                        neuronConnectionsFrom(fromNeuron);

                foreach (Connection *c, connections) {
                    if (!toLayer->contains(c->destination())) {
                        continue;
                    }

                    int j = toLayer->neurons.indexOf(c->destination());
                    output[j] +=
                            *(neuronConnection(fromNeuron, c->destination()))
                            * input.at(i);
                }
            }

            return output;
        }


        ValueVector NeuralNetwork::calculate(const ValueVector &input)
                throw(LayerSizeMismatchException)
        {
            if (input.size() != m_layers.first()->size()) {
                throw LayerSizeMismatchException(
                        m_layers.first()->size(),
                        input.size());
            }

            return m_pattern->calculate(this, input);
        }


        QTextStream& operator<<(QTextStream &out, const NeuralNetwork &network)
        {
            QVariantMap outList;

            outList.insert("Version", NeuralNetwork::VERSION);

            QList<QVariant> layersList;

            outList.insert("Layers", layersList);

            for (int i = 0; i != network.m_layers.size(); ++i) {
                QVariantMap layerMap;
                QVariantList neuronsList;

                QList<Neuron*> neurons = network.m_layers[i]->neurons;
                for (int j = 0; j != neurons.size(); ++j) {
                    QVariantMap neuronMap;

                    neuronMap.insert(
                            "ActivationFunction",
                            neurons[j]->m_activationFunction
                                ->metaObject()->className());
                    neuronMap.insert(
                            "LastResult",
                            neurons[j]->lastResult());

                    neuronsList.append(neuronMap);
                }

                layerMap.insert("Neurons", neuronsList);
                layersList.append(layerMap);
            }

            outList.insert("Layers", layersList);

            QVariantList connections;

            for (int i = 0; i != network.size(); ++i) {

                for (int j = 0; j != network[i]->size() + 1; ++j) {
                    Neuron *srcNeuron = network.layerAt(i)->neuronAt(j);

                    if (!network.m_connectionSources.contains(srcNeuron)) {
                        continue;
                    }

                    QList<Connection*> networkConnections =
                            network.m_connectionSources[srcNeuron];

                    // Find destination neurons:

                    foreach (Connection *c, networkConnections) {
                        Neuron *dstNeuron = c->destination();

                        for (int k = 0; k != network.size(); ++k) {
                            if (!network[k]->contains(dstNeuron)) {
                                continue;
                            }


                            QVariantMap connection;
                            connection.insert("srcLayer", i);
                            connection.insert("srcNeuron", j);
                            connection.insert("dstLayer", k);
                            connection.insert(
                                    "dstNeuron",
                                    network[k]->neurons.indexOf(dstNeuron));
                            connection.insert("weight", c->weight());
                            connection.insert("fixed", c->fixedWeight());
                            connections.append(connection);

                            break;
                        }
                    }
                }
            }

            outList.insert("Connections", connections);

            // Serialize:

            QJson::Serializer serializer;
            serializer.setIndentMode(QJson::IndentFull);

            bool ok;
            QByteArray json = serializer.serialize(outList, &ok);

            Q_ASSERT(ok);
            if (!ok) {
                return out;
            }

            out << json;
            return out;
        }
    }
}
