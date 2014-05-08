/*!
 * \file	NeuralNetwork.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include <QObject>

#include <QList>
#include <QVector>

#include <QByteArray>
#include <QTextStream>

#include <qjson/serializer.h>
#include <qjson/qobjecthelper.h>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include <functional>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "Exception.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "TrainingAlgorithm.h"


using std::function;


namespace Winzent {
    namespace ANN {

        const char NeuralNetwork::VERSION[] = "1.0";


        log4cxx::LoggerPtr NeuralNetwork::logger =
                log4cxx::LogManager::getLogger("Winzent.ANN.NeuralNetwork");


        NeuralNetwork::NeuralNetwork(QObject* parent):
                QObject(parent),
                m_pattern(nullptr)
        {
            Q_ASSERT(m_connectionSources.size()
                     == m_connectionDestinations.size());
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork &rhs):
                QObject(rhs.parent()),
                m_pattern(nullptr)
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

                // Ignore leftovers:

                if (rhs.neuronConnectionsFrom(foreignNeuron).size() == 0) {
                    continue;
                }

                // First, find the index of the source neuron:

                int srcLayerIndex = -1;
                int srcNeuronIndex = -1;

                for (int i = 0; i != rhs.size(); ++i) {
                    if (!rhs[i]->contains(foreignNeuron)) {
                        continue;
                    }

                    srcLayerIndex = i;
                    srcNeuronIndex = rhs[i]->indexOf(foreignNeuron);
                    break;
                }

                Q_ASSERT(srcLayerIndex >= 0 && srcNeuronIndex >= 0);

                // Then, find the index of the destination neurons:

                QList<Connection*> connections =
                        rhs.neuronConnectionsFrom(foreignNeuron);

                Q_ASSERT(connections.size() > 0);

                foreach (Connection *c, connections) {
                    int dstLayerIndex = -1;
                    int dstNeuronIndex = -1;

                    for (int i = 0; i != rhs.size(); ++i) {
                        if (!rhs[i]->contains(c->destination())) {
                            continue;
                        }

                        dstLayerIndex = i;
                        dstNeuronIndex = rhs[i]->indexOf(c->destination());
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


        NeuralNetwork *NeuralNetwork::clone() const
        {
            return new NeuralNetwork(*this);
        }


        bool NeuralNetwork::containsNeuron(const Neuron *neuron) const
        {
            Q_ASSERT(nullptr != neuron);

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
                const Neuron *const &from,
                const Neuron *const &to)
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
                const Neuron *const &from,
                const Neuron *const &to,
                double value)
                    throw(NoConnectionException, WeightFixedException)
        {
            neuronConnection(from, to)->weight(value);
        }


        const QList<Connection *> NeuralNetwork::neuronConnectionsFrom(
                const Neuron *const &neuron)
                    const
                    throw(UnknownNeuronException)
        {
            Q_ASSERT(NULL != neuron);

            if (!containsNeuron(neuron)) {
                throw UnknownNeuronException(neuron);
            }

            return m_connectionSources[const_cast<Neuron*>(neuron)];
        }


        QList<Connection*> NeuralNetwork::neuronConnectionsTo(
                const Neuron *const &neuron)
                    const
                    throw(UnknownNeuronException)
        {
            Q_ASSERT(NULL != neuron);

            if (!containsNeuron(neuron)) {
                throw UnknownNeuronException(neuron);
            }

            return m_connectionDestinations[const_cast<Neuron*>(neuron)];
        }


        void NeuralNetwork::eachLayer(
                std::function<void (const Layer *const &)> yield) const
        {
            std::for_each(m_layers.begin(), m_layers.end(), yield);
        }


        void NeuralNetwork::eachLayer(function<void (Layer* const &)> yield)
        {
            std::for_each(m_layers.begin(), m_layers.end(), yield);
        }


        void NeuralNetwork::eachConnection(
                std::function<void (Connection * const &)> yield)
        {
            eachLayer([this, &yield](Layer *const &layer) {
                layer->eachNeuron([this, &yield](Neuron *const &neuron) {
                    QList<Connection *> connections =
                            neuronConnectionsFrom(neuron);
                    std::for_each(
                            connections.begin(),
                            connections.end(),
                            yield);
                });

                // Extra for the bias neuron:

                QList<Connection *> connections = neuronConnectionsFrom(
                        layer->biasNeuron());
                std::for_each(
                        connections.begin(),
                        connections.end(),
                        yield);
            });
        }


        void NeuralNetwork::eachConnection(
                std::function<void (const Connection *const &)> yield) const
        {
            eachLayer([this, &yield](const Layer *const &layer) {
                layer->eachNeuron([this, &yield](const Neuron *const &neuron) {
                    QList<Connection *> connections =
                            neuronConnectionsFrom(neuron);
                    std::for_each(
                            connections.begin(),
                            connections.end(),
                            yield);
                });

                // Extra for the bias neuron:

                QList<Connection *> connections = neuronConnectionsFrom(
                        layer->biasNeuron());
                std::for_each(
                        connections.begin(),
                        connections.end(),
                        yield);
            });
        }


        Connection *NeuralNetwork::connectNeurons(
                Neuron *const &from,
                Neuron *const &to)
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

            m_connectionSources[from] << connection;
            m_connectionDestinations[to] << connection;

            return connection;
        }


        NeuralNetwork &NeuralNetwork::operator<<(Layer *layer)
        {
            m_layers << layer;
            layer->setParent(this);

            // Make sure the bias connection exists:

            Neuron *bias = layer->biasNeuron();

            for (int i = 0; i != layer->size(); ++i) {
                Connection *c = connectNeurons(bias, layer->neuronAt(i));
                c->weight(-1.0);
            }

            return *this;
        }


        Layer *&NeuralNetwork::layerAt(const int &index)
        {
            return m_layers[index];
        }


        Layer *const &NeuralNetwork::layerAt(const int &index) const
        {
            return m_layers.at(index);
        }


        Layer *&NeuralNetwork::operator [](const int &index)
        {
            return layerAt(index);
        }


        Layer *NeuralNetwork::operator [](const int &index) const
        {
            return layerAt(index);
        }


        Layer *const &NeuralNetwork::inputLayer() const
        {
            return m_layers.first();
        }


        Layer *const &NeuralNetwork::outputLayer() const
        {
            return m_layers.last();
        }


        void NeuralNetwork::configure(const NeuralNetworkPattern *pattern)
        {
            // Get rid of the old pattern, if one exists:

            if (nullptr != m_pattern) {
                delete m_pattern;
            }

            m_pattern = pattern->clone();
            m_pattern->setParent(this);

            m_pattern->configureNetwork(this);
        }


        ValueVector NeuralNetwork::calculateLayer(
                Layer *const &layer,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            if (layer->size() != input.size()) {
                throw LayerSizeMismatchException(input.size(), layer->size());
            }

            ValueVector output;

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

                output << neuron->activate(sum);
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

                    Q_ASSERT(c->source() == fromNeuron);

                    int j = toLayer->indexOf(c->destination());
                    output[j] += input.at(i) * c->weight();
                }
            }

            LOG4CXX_DEBUG(
                    logger,
                    "ANN Layer Transition: "
                        << from << ": " << input
                        << " => "
                        << to << ": " << output);
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

            outList.insert("version", NeuralNetwork::VERSION);

            QList<QVariant> layersList;

            outList.insert("layers", layersList);

            for (int i = 0; i != network.m_layers.size(); ++i) {
                QVariantMap layerMap;
                QVariantList neuronsList;

                for (int j = 0; j != network.layerAt(i)->size() + 1; ++j) {
                    QVariantMap neuronMap;

                    neuronMap.insert(
                            "activationFunction",
                            network.layerAt(i)->neuronAt(j)
                                ->m_activationFunction
                                    ->metaObject()->className());

                    QVariantList lastInputs;
                    foreach (qreal r,
                             network.layerAt(i)->neuronAt(j)->lastInputs()) {
                        lastInputs << r;
                    }

                    neuronMap.insert(
                            "lastInputs",
                            lastInputs);

                    QVariantList lastResults;
                    foreach (qreal r,
                             network.layerAt(i)->neuronAt(j)->lastResults()) {
                        lastResults << r;
                    }

                    neuronMap.insert(
                            "lastResults",
                            lastResults);

                    neuronsList.append(neuronMap);
                }

                layerMap.insert("neurons", neuronsList);
                layersList.append(layerMap);
            }

            outList.insert("layers", layersList);

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
                                    network[k]->indexOf(dstNeuron));
                            connection.insert("weight", c->weight());
                            connection.insert("fixed", c->fixedWeight());
                            connections.append(connection);

                            break;
                        }
                    }
                }
            }

            outList.insert("connections", connections);

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


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::ValueVector &valueVector)
    {
        os << "ValueVector(";

        for (int i = 0; i < valueVector.size(); ++i) {
            os << valueVector.at(i);
            if (i < valueVector.size() - 1) {
                os << ", ";
            }
        }

        os << ")";
        return os;
    }
}
