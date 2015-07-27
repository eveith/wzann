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

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>

#include <memory>
#include <cstddef>
#include <functional>

#include <boost/ptr_container/ptr_vector.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "Exception.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "ConstantActivationFunction.h"

#include "TrainingAlgorithm.h"


using std::function;


namespace Winzent {
    namespace ANN {

        const char NeuralNetwork::VERSION[] = "1.0";


        log4cxx::LoggerPtr NeuralNetwork::logger =
                log4cxx::LogManager::getLogger("Winzent.ANN.NeuralNetwork");


        NeuralNetwork::NeuralNetwork(QObject* parent):
                QObject(parent),
                m_biasNeuron(new Neuron(new ConstantActivationFunction()))
        {
            Q_ASSERT(m_connectionSources.size()
                     == m_connectionDestinations.size());
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork &rhs):
                QObject(rhs.parent()),
                m_biasNeuron(rhs.m_biasNeuron->clone())
        {
            // Clone layers:

            for (const Layer &l: rhs.m_layers) {
                Layer *layerClone = l.clone();
                *this << layerClone;
            }

            // Clone connections. We have already cloned the neurons, which
            // means that we cannot simply clone all connections because we'd
            // end up with pointers to the original. Instead, we need to map
            // via the index in the layer and create connections of our own:

            foreach (Neuron *foreignNeuron, rhs.m_connectionSources.keys()) {

                // Ignore unconnected neurons:

                if (rhs.neuronConnectionsFrom(foreignNeuron).size() == 0) {
                    continue;
                }

                QList<Connection *> connections;
                int srcLayerIndex   = -1;
                int srcNeuronIndex  = -1;

                if (rhs.biasNeuron() == foreignNeuron) {
                    connections = rhs.neuronConnectionsFrom(foreignNeuron);
                } else {
                    // First, find the index of the source neuron:

                    for (int i = 0; i != rhs.size(); ++i) {
                        if (!rhs[i]->contains(foreignNeuron)) {
                            continue;
                        }

                        srcLayerIndex = i;
                        srcNeuronIndex = rhs[i]->indexOf(foreignNeuron);
                        break;
                    }

                    Q_ASSERT(srcLayerIndex >= 0 && srcNeuronIndex >= 0);

                    connections = rhs.neuronConnectionsFrom(foreignNeuron);
                }

                Q_ASSERT(connections.size() > 0);

                // Find the index of the destination neurons and re-create:

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

                    Neuron *srcNeuron = (rhs.biasNeuron() == foreignNeuron
                            ? biasNeuron()
                            : layerAt(srcLayerIndex)->neuronAt(srcNeuronIndex));
                    Neuron *dstNeuron =
                            layerAt(dstLayerIndex)->neuronAt(dstNeuronIndex);

                    Connection *newConnection = connectNeurons(
                            srcNeuron,
                            dstNeuron);
                    newConnection->weight(c->weight());
                    newConnection->fixedWeight(c->fixedWeight());
                }
            }

            // Make sure the cloned pattern has the correct parent object:

            if (nullptr != rhs.m_pattern) {
                m_pattern.reset(rhs.m_pattern->clone());
            }
        }


        NeuralNetwork::~NeuralNetwork()
        {
            delete m_biasNeuron;
        }


        NeuralNetwork *NeuralNetwork::clone() const
        {
            return new NeuralNetwork(*this);
        }


        const Neuron *NeuralNetwork::biasNeuron() const
        {
            return m_biasNeuron;
        }


        Neuron *NeuralNetwork::biasNeuron()
        {
            return m_biasNeuron;
        }


        bool NeuralNetwork::containsNeuron(const Neuron *const &neuron) const
        {
            Q_ASSERT(nullptr != neuron);

            if (biasNeuron() == neuron) {
                return true;
            }

            return std::any_of(m_layers.begin(), m_layers.end(), [&neuron](
                    const Layer &layer) {
                return layer.contains(neuron);
            });
        }


        bool NeuralNetwork::neuronConnectionExists(
                const Neuron *const &from,
                const Neuron *const &to) const
        {
            Q_ASSERT(from != nullptr);
            Q_ASSERT(to != nullptr);

            if (! m_connectionSources.contains(const_cast<Neuron *>(from))) {
                return false;
            }

            const QList<Connection *> &connections = m_connectionSources[
                    const_cast<Neuron *>(from)];

            return std::any_of(connections.begin(), connections.end(), [&to](
                    const Connection *const &c) {
                return (c->destination() == const_cast<Neuron *>(to));
            });
        }


        Connection *NeuralNetwork::neuronConnection(
                const Neuron *const &from,
                const Neuron *const &to)
                    const
                    throw(NoConnectionException)
        {
            if (!neuronConnectionExists(from, to)) {
                throw NoConnectionException();
            }

            Connection *result = nullptr;

            QList<Connection*> connections =
                    m_connectionSources[const_cast<Neuron*>(from)];
            foreach(Connection *c, connections) {
                if (c->destination() == const_cast<Neuron*>(to)) {
                    result = c;
                    break;
                }
            }

            return result;
        }

        const QList<Connection *> NeuralNetwork::neuronConnectionsFrom(
                const Neuron *const &neuron)
                    const
                    throw(UnknownNeuronException)
        {
            Q_ASSERT(nullptr != neuron);

            if (!containsNeuron(neuron)) {
                throw UnknownNeuronException(neuron);
            }

            return m_connectionSources[const_cast<Neuron*>(neuron)];
        }


        QList<Connection *> NeuralNetwork::neuronConnectionsTo(
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
            std::for_each(m_layers.begin(), m_layers.end(),
                    [&yield](const Layer &l) {
                yield(&l);
            });
        }


        void NeuralNetwork::eachLayer(function<void (Layer* const &)> yield)
        {
            std::for_each(m_layers.begin(), m_layers.end(),
                    [&yield](Layer &l) {
                yield(&l);
            });
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
            });

            QList<Connection *> biasNeuronConnections =
                    neuronConnectionsFrom(biasNeuron());
            std::for_each(
                    biasNeuronConnections.begin(),
                    biasNeuronConnections.end(),
                    yield);
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
            });

            QList<Connection *> biasNeuronConnections =
                    neuronConnectionsFrom(biasNeuron());
            std::for_each(
                    biasNeuronConnections.begin(),
                    biasNeuronConnections.end(),
                    yield);

        }


        Connection *NeuralNetwork::connectNeurons(
                Neuron *const &from,
                Neuron *const &to)
                    throw(UnknownNeuronException)
        {
            if (from != biasNeuron() && !containsNeuron(from)) {
                throw UnknownNeuronException(from);
            }

            if (!containsNeuron(to)) {
                throw UnknownNeuronException(to);
            }

            Connection *connection = new Connection(from, to, 0.0);

            Q_ASSERT(connection->source() == from);
            Q_ASSERT(connection->destination() == to);

            m_connectionSources[from] << connection;
            m_connectionDestinations[to] << connection;

            return connection;
        }


        NeuralNetwork &NeuralNetwork::operator <<(Layer *const &layer)
        {
            layer->m_parent = this;
            m_layers.push_back(layer);

            // Connect all neurons to the bias neuron, but only if the new
            // layer is not the input layer.

            if (&(m_layers.front()) != layer) {
                for (Neuron &n: *layer) {
                    connectNeurons(biasNeuron(), &n)->weight(-1.0);
                }
            }

            return *this;
        }


        Layer *NeuralNetwork::layerAt(const size_t &index) const
        {
            return &(const_cast<NeuralNetwork *>(this)->m_layers.at(index));
        }


        Layer *NeuralNetwork::operator [](const size_t &index) const
        {
            return layerAt(index);
        }


        Layer &NeuralNetwork::operator [](const size_t &index)
        {
            return m_layers[index];
        }


        Layer *NeuralNetwork::inputLayer() const
        {
            return &(const_cast<NeuralNetwork *>(this)->m_layers.front());
        }


        Layer *NeuralNetwork::outputLayer() const
        {
            return &(const_cast<NeuralNetwork *>(this)->m_layers.back());
        }


        NeuralNetwork &NeuralNetwork::configure(
                const NeuralNetworkPattern &pattern)
        {
            m_pattern.reset(pattern.clone());
            m_pattern->configureNetwork(this);
            return *this;
        }


        NeuralNetwork &NeuralNetwork::configure(
                const NeuralNetworkPattern *const &pattern)
        {
            Q_ASSERT(nullptr != pattern);
            return configure(*pattern);
        }


        ValueVector NeuralNetwork::calculateLayer(
                Layer *const &layer,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
#ifdef QT_DEBUG
            if (layer->size() != input.size()) {
                throw LayerSizeMismatchException(input.size(), layer->size());
            }
#endif

            ValueVector output;
            output.reserve(layer->size());

            for (int i = 0; i != input.size(); ++i) {
                double sum = input.at(i);
                Neuron *neuron = layer->neuronAt(i);

                // Add bias neuron. We ignore the bias neuron in the input layer
                // even when it's there; it does not make sense to include the
                // bias neuron in the input layer since its output would be
                // overwritten by the input anyways.

                if (inputLayer() != layer
                        && neuronConnectionExists(biasNeuron(), neuron)) {
                    sum += neuronConnection(biasNeuron(), neuron)->weight()
                            * biasNeuron()->activate(1.0);
                }

                output << neuron->activate(sum);
            }

            return (output);
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

#ifdef QT_DEBUG
            if (input.size() != fromLayerSize) {
                throw LayerSizeMismatchException(input.size(), fromLayerSize);
            }
#endif

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
            return (output);
        }


        ValueVector NeuralNetwork::calculate(const ValueVector &input)
                throw(LayerSizeMismatchException)
        {
            if (input.size() != m_layers.front().size()) {
                throw LayerSizeMismatchException(
                        m_layers.front().size(),
                        input.size());
            }

            return m_pattern->calculate(this, input);
        }


        QTextStream &operator<<(QTextStream &out, const NeuralNetwork &network)
        {
            QJsonDocument jsonDocument;
            QJsonObject outList;

            outList.insert("version", QString(NeuralNetwork::VERSION));

            QJsonArray layersList;

            for (int i = 0; i != network.m_layers.size(); ++i) {
                QJsonObject layerMap;
                QJsonArray neuronsList;

                for (int j = 0; j != network.layerAt(i)->size(); ++j) {
                    QJsonObject neuronMap;

                    neuronMap.insert(
                            "activationFunction",
                            "unknown: TODO FIXME!");

                    QJsonArray lastInputs;
                    foreach (double r,
                             network.layerAt(i)->neuronAt(j)->lastInputs()) {
                        lastInputs.append(r);
                    }

                    neuronMap.insert("lastInputs", lastInputs);

                    QJsonArray lastResults;
                    foreach (double r,
                             network.layerAt(i)->neuronAt(j)->lastResults()) {
                        lastResults.append(r);
                    }

                    neuronMap.insert("lastResults", lastResults);
                    neuronsList.append(neuronMap);
                }

                layerMap.insert("neurons", neuronsList);
                layersList.append(layerMap);
            }

            outList.insert("layers", layersList);

            QJsonArray connections;

            for (int i = 0; i != network.size(); ++i) {
                for (int j = 0; j != network[i]->size(); ++j) {
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


                            QJsonObject connection;
                            connection.insert("srcLayer", i);
                            connection.insert("srcNeuron", j);
                            connection.insert("dstLayer", k);
                            connection.insert(
                                    "dstNeuron",
                                    static_cast<qint64>(
                                        network[k]->indexOf(dstNeuron)));
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

            jsonDocument.setObject(outList);
            out << jsonDocument.toJson();
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
