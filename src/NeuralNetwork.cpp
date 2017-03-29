#include <QByteArray>
#include <QTextStream>

#include <QJsonValue>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <memory>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <unordered_map>

#include <boost/range.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include <ClassRegistry.h>
#include <JsonSerializable.h>

#include "QtContainerOutput.h"

#include "Layer.h"
#include "Neuron.h"
#include "Exception.h"
#include "Connection.h"
#include "NeuralNetworkPattern.h"

#include "ActivationFunction.h"
#include "ConstantActivationFunction.h"

#include "NeuralNetwork.h"
#include "Winzent-ANN_global.h"


using std::function;
using log4cxx::LogManager;
using boost::make_iterator_range;


namespace Winzent {
    namespace ANN {

        const char NeuralNetwork::VERSION[] = "1.0";


        NeuralNetwork::NeuralNetwork():
                logger(LogManager::getLogger("Winzent.ANN.NeuralNetwork")),
                m_biasNeuron(new Neuron(new ConstantActivationFunction()))
        {
            Q_ASSERT(m_connectionSources.size()
                     == m_connectionDestinations.size());
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork &rhs):
                logger(LogManager::getLogger("Winzent.ANN.NeuralNetwork")),
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

            for (auto connectionSources: rhs.m_connectionSources) {
                auto foreignNeuron = connectionSources.first;

                ConnectionConstRange connections;
                int srcLayerIndex   = -1;
                int srcNeuronIndex  = -1;

                if (&(rhs.biasNeuron()) == foreignNeuron) {
                    connections = rhs.connectionsFrom(*foreignNeuron);
                } else {
                    // First, find the index of the source neuron:

                    for (size_type i = 0; i != rhs.size(); ++i) {
                        if (! rhs[i].contains(*foreignNeuron)) {
                            continue;
                        }

                        srcLayerIndex = i;
                        srcNeuronIndex = rhs[i].indexOf(*foreignNeuron);
                        break;
                    }

                    Q_ASSERT(srcLayerIndex >= 0 && srcNeuronIndex >= 0);

                    connections = rhs.connectionsFrom(*foreignNeuron);
                }

                Q_ASSERT(connections.second-connections.first > 0);

                // Find the index of the destination neurons and re-create:

                for (Connection *c: make_iterator_range(connections)) {
                    int dstLayerIndex = -1;
                    int dstNeuronIndex = -1;

                    for (size_type i = 0; i != rhs.size(); ++i) {
                        if (!rhs[i].contains(c->destination())) {
                            continue;
                        }

                        dstLayerIndex = i;
                        dstNeuronIndex = rhs[i].indexOf(c->destination());
                        break;
                    }

                    Q_ASSERT(dstLayerIndex >= 0 && dstNeuronIndex >= 0);

                    // Now, re-create the connection here:

                    auto &srcNeuron = (&(rhs.biasNeuron()) == foreignNeuron
                            ? biasNeuron()
                            : (*this)[srcLayerIndex][srcNeuronIndex]);
                    auto &dstNeuron = (*this)[dstLayerIndex][dstNeuronIndex];

                    auto &newConnection = connectNeurons(
                            srcNeuron,
                            dstNeuron);
                    newConnection.weight(c->weight());
                    newConnection.fixedWeight(c->fixedWeight());
                }
            }

            // Make sure the cloned pattern has the correct parent object:

            if (nullptr != rhs.m_pattern) {
                m_pattern.reset(rhs.m_pattern->clone());
            }
        }


        NeuralNetwork::~NeuralNetwork()
        {
            clear();
        }


        NeuralNetwork *NeuralNetwork::clone() const
        {
            return new NeuralNetwork(*this);
        }


        const Neuron &NeuralNetwork::biasNeuron() const
        {
            return *m_biasNeuron;
        }


        Neuron &NeuralNetwork::biasNeuron()
        {
            return *m_biasNeuron;
        }


        bool NeuralNetwork::contains(const Neuron &neuron) const
        {
            if (&(biasNeuron()) == &neuron) {
                return true;
            }

            return std::any_of(m_layers.begin(), m_layers.end(), [&neuron](
                    const Layer &layer) {
                return layer.contains(neuron);
            });
        }


        bool NeuralNetwork::connectionExists(
                const Neuron &from,
                const Neuron &to)
                const
        {
            auto it = m_connectionDestinations.find(
                    const_cast<Neuron *>(&to));
            if (m_connectionDestinations.end() == it) {
                return false;
            }

            return std::any_of(
                    it->second.begin(),
                    it->second.end(),
                    [&from](const Connection *const &c) {
                return (c->source() == from);
            });
        }


        Connection *NeuralNetwork::connection(
                const Neuron &from,
                const Neuron &to)
        {
            auto connections = connectionsTo(to);
            auto cit = std::find_if(
                    connections.first,
                    connections.second,
                    [&from](const Connection *const &c) {
                return c->source() == from;
            });

            if (cit != connections.second) {
                return *cit;
            }

            throw NoConnectionException();
            return nullptr;
        }


        NeuralNetwork::ConnectionConstRange
        NeuralNetwork::connectionsFrom(const Neuron &neuron) const
        {
            auto *n = const_cast<Neuron *>(&neuron);
            auto it = m_connectionSources.find(n);

            if (m_connectionSources.end() == it) {
                return std::make_pair(
                        ConnectionsVector::const_iterator(),
                        ConnectionsVector::const_iterator());
            } else {
                return std::make_pair(
                        m_connectionSources.at(n).begin(),
                        m_connectionSources.at(n).end());
            }
        }


        NeuralNetwork::ConnectionRange
        NeuralNetwork::connectionsFrom(const Neuron &neuron)
        {
            auto *n = const_cast<Neuron *>(&neuron);
            auto it = m_connectionSources.find(n);

            if (m_connectionSources.end() == it) {
                return std::make_pair(
                        ConnectionsVector::iterator(),
                        ConnectionsVector::iterator());
            } else {
                return std::make_pair(
                        m_connectionSources.at(n).begin(),
                        m_connectionSources.at(n).end());
            }
        }


        NeuralNetwork::ConnectionConstRange
        NeuralNetwork::connectionsTo(const Neuron &neuron) const
        {
            auto *n = const_cast<Neuron *>(&neuron);
            auto it = m_connectionDestinations.find(n);

            if (m_connectionDestinations.end() == it) {
                return std::make_pair(
                        ConnectionsVector::const_iterator(),
                        ConnectionsVector::const_iterator());
            } else {
                return std::make_pair(
                        m_connectionDestinations.at(n).begin(),
                        m_connectionDestinations.at(n).end());
            }
       }


        NeuralNetwork::ConnectionRange
        NeuralNetwork::connectionsTo(const Neuron &neuron)
        {
            auto *n = const_cast<Neuron *>(&neuron);
            auto it = m_connectionDestinations.find(n);

            if (m_connectionDestinations.end() == it) {
                return std::make_pair(
                        ConnectionsVector::iterator(),
                        ConnectionsVector::iterator());
            } else {
                return std::make_pair(
                        m_connectionDestinations.at(n).begin(),
                        m_connectionDestinations.at(n).end());
            }
        }


        Connection &NeuralNetwork::connectNeurons(
                const Neuron &from,
                const Neuron &to)
        {
            if (&from != &biasNeuron() && ! contains(from)) {
                throw UnknownNeuronException(&from);
            }

            if (! contains(to)) {
                throw UnknownNeuronException(&to);
            }

            Neuron &src = const_cast<Neuron &>(from),
                    &dst = const_cast<Neuron &>(to);

            Connection *connection = new Connection(src, dst, 0.0);

            m_connectionSources[&src].push_back(connection);
            m_connectionDestinations[&dst].push_back(connection);

            return *connection;
        }


        void NeuralNetwork::disconnectNeurons(
                const Neuron &from,
                const Neuron &to)
        {
            auto &sources = m_connectionSources[const_cast<Neuron *>(&from)];
            auto connectionIt = std::find_if(
                    sources.begin(),
                    sources.end(),
                    [&from, &to](Connection *connection) {
                return connection->source() == from
                        && connection->destination() == to;
            });

            auto &destinations = m_connectionDestinations.at(
                    const_cast<Neuron *>(&to));
            std::remove_if(
                    destinations.begin(),
                    destinations.end(),
                    [&from, &to](Connection *connection) {
                return connection->source() == from
                        && connection->destination() == to;
            });

            if (connectionIt != sources.end()) {
                sources.erase(connectionIt);
                delete *connectionIt;
            }
        }


        NeuralNetwork &NeuralNetwork::operator <<(Layer *layer)
        {
            layer->m_parent = this;
            m_layers.push_back(layer);
            return *this;
        }


        Layer *NeuralNetwork::layerAt(const size_type &index) const
        {
            return &(const_cast<NeuralNetwork *>(this)->m_layers.at(index));
        }


        const Layer& NeuralNetwork::operator [](const size_type &index) const
        {
            return m_layers[index];
        }


        Layer &NeuralNetwork::operator [](const size_type &index)
        {
            return m_layers[index];
        }


        Layer &NeuralNetwork::inputLayer()
        {
            return m_layers.front();
        }


        Layer &NeuralNetwork::outputLayer()
        {
            return m_layers.back();
        }


        std::pair<
                NeuralNetwork::LayerConstIterator,
                NeuralNetwork::LayerConstIterator>
        NeuralNetwork::layers() const
        {
            return std::make_pair(m_layers.begin(), m_layers.end());
        }


        std::pair<NeuralNetwork::LayerIterator, NeuralNetwork::LayerIterator>
        NeuralNetwork::layers()
        {
            return std::make_pair(m_layers.begin(), m_layers.end());
        }


        NeuralNetwork &NeuralNetwork::configure(
                const NeuralNetworkPattern &pattern)
        {
            m_pattern.reset(pattern.clone());
            m_pattern->configureNetwork(*this);
            return *this;
        }


        Vector NeuralNetwork::calculateLayerTransition(
                const Layer &from,
                const Layer &to,
                const Vector &input)
        {

#ifdef QT_DEBUG
            auto fromLayerSize= from.size();
            if (static_cast<Layer::size_type>(input.size()) != fromLayerSize){
                throw LayerSizeMismatchException(input.size(), fromLayerSize);
            }
#endif

            Vector output;
            auto toLayerSize = to.size();
            output.fill(0.0, toLayerSize);

            for (Layer::size_type t = 0; t != toLayerSize; ++t) {
                const Neuron &toNeuron = to[t];
                auto connections = connectionsTo(toNeuron);

                for (const auto &c: make_iterator_range(connections)) {
                    Q_ASSERT(&(c->destination()) == &toNeuron);

                    if (! from.contains(c->source())) {
                        continue;
                    }

                    auto s = from.indexOf(c->source());
                    output[t] += input.at(s) * c->weight();
                }
            }

            return output;
        }


        Vector NeuralNetwork::calculateLayer(
                Layer &layer,
                const Vector &input)
        {
            Vector biasedInput(input.size());

            auto it = layer.begin();
            for (Vector::size_type i = 0; i != input.size(); ++i, it++) {
                auto &neuron = *it;
                biasedInput[i] = input[i];

                if (! connectionExists(biasNeuron(), neuron)) {
                    continue;
                }

                const auto &c = connection(biasNeuron(), neuron);
                biasedInput[i] += biasNeuron().activate(1.0) * c->weight();
            }

            Q_ASSERT(it == layer.end());
            return layer.activate(biasedInput);
        }


        Vector NeuralNetwork::calculate(const Vector &input)
        {
            if (static_cast<Layer::size_type>(input.size())
                    != m_layers.front().size()) {
                throw LayerSizeMismatchException(
                        m_layers.front().size(),
                        input.size());
            }

            return m_pattern->calculate(*this, input);
        }


        void NeuralNetwork::clear()
        {
            for (auto &destinations: m_connectionDestinations) {
                for (auto &connection: destinations.second) {
                    delete connection;
                }
            }

            m_pattern.reset(nullptr);
            m_layers.clear();
            m_connectionSources.clear();
            m_connectionDestinations.clear();
        }


        void NeuralNetwork::fromJSON(const QJsonDocument &json)
        {
            clear();
            QJsonObject o = json.object();

            m_biasNeuron->fromJSON(QJsonDocument(o["biasNeuron"].toObject()));

            QJsonArray layers = o["layers"].toArray();
            for (const auto &i: layers) {
                Layer *l = new Layer();
                l->fromJSON(QJsonDocument(i.toArray()));
                *this << l;
            }

            QJsonArray connections = o["connections"].toArray();
            for (const auto &i: connections) {
                QJsonObject c = i.toObject();

                auto &src = (c["srcNeuron"] == "BIAS"
                        ? biasNeuron()
                        : (*this)[c["srcLayer"].toInt()][
                            c["srcNeuron"].toInt()]);
                auto &dst = (*this)[c["dstLayer"].toInt()][
                        c["dstNeuron"].toInt()];

                connectNeurons(src, dst)
                    .weight(c["weight"].toDouble())
                    .fixedWeight(c["fixedWeight"].toBool());
            }

            if (o.contains("pattern") && o["pattern"].isObject()) {
                QJsonObject pattern = o["pattern"].toObject();
                m_pattern.reset(
                        ClassRegistry<NeuralNetworkPattern>::instance()
                            ->create(pattern["type"].toString()));
                Q_ASSERT(m_pattern != nullptr);
                m_pattern->fromJSON(QJsonDocument(pattern));
            }
        }


        QJsonDocument NeuralNetwork::toJSON() const
        {
            QJsonObject o;

            o["version"] = VERSION;
            o["biasNeuron"] = m_biasNeuron->toJSON().object();

            QJsonArray layers;
            for (const Layer &l: m_layers) {
                layers.push_back(l.toJSON().array());
            }
            o["layers"] = layers;

            QJsonArray connections;
            for (const auto &connectionSources: m_connectionSources) {
                for (const auto &c: connectionSources.second) {
                    int srcLayer = -1,
                            dstLayer = -1,
                            srcNeuron = -1,
                            dstNeuron = -1;
                    QJsonObject connection;

                    for (size_t i = 0; i != m_layers.size(); ++i) {
                        for (size_t j = 0; j != m_layers.at(i).size(); ++j) {
                            const auto &n = layerAt(i)->neuronAt(j);

                            if (n == &(c->source())) {
                                srcLayer = i;
                                srcNeuron = j;
                            } else if (n == &(c->destination())) {
                                dstLayer = i;
                                dstNeuron = j;
                            }
                        }
                    }

                    connection["srcLayer"] = srcLayer;
                    connection["srcNeuron"] = srcNeuron;
                    connection["dstLayer"] = dstLayer;
                    connection["dstNeuron"] = dstNeuron;
                    connection["weight"] = c->weight();
                    connection["fixedWeight"] = c->fixedWeight();

                    if (-1 == srcNeuron) {
                        connection["srcNeuron"] = "BIAS";
                    }

                    connections.push_back(connection);
                }
            }
            o["connections"] = connections;

            o["pattern"] = QJsonValue::Null;
            if (m_pattern != nullptr) {
                o["pattern"] = m_pattern->toJSON().object();
            }

            return QJsonDocument(o);
        }


        bool NeuralNetwork::operator ==(const NeuralNetwork &other) const
        {
            bool equal = true;

            equal &= size() == other.size();
            equal &= biasNeuron().equals(other.biasNeuron());
            equal &= ((m_pattern == nullptr && other.m_pattern == nullptr)
                    || (m_pattern != nullptr && other.m_pattern != nullptr
                        && m_pattern->equals(other.m_pattern.get())));
            equal &= (m_connectionSources.size()
                    == other.m_connectionSources.size());

            if (! equal) { // Short cut in order to save time:
                return equal;
            }

            auto lit1 = m_layers.begin(), lit2 = other.m_layers.begin();
            for (; lit1 != m_layers.end() && lit2 != other.m_layers.end();
                    lit1++, lit2++) {
                if (! (*lit1 == *lit2)) {
                    return false;
                }
            }

            for (size_type i = 0; i != size(); ++i) {
                const auto& layer = (*this)[i];

                for (Layer::size_type j = 0; j != layer.size(); ++j) {
                    const auto& ourConnections = connectionsTo(layer[j]);
                    const auto& otherConnections =
                            other.connectionsTo(other[i][j]);

                    for (auto it1 = ourConnections.first,
                                it2 = otherConnections.first;
                            it1 != ourConnections.second
                                && it2 != otherConnections.second;
                            it1++, it2++) {
                        if (! (*it1)->equals(**it2)) {
                            return false;
                        }
                    }
                }
            }


            return equal;
        }


        bool NeuralNetwork::operator !=(const NeuralNetwork& other) const
        {
            return !(*this == other);
        }
    }
}


QTextStream &operator<<(
        QTextStream &out,
        Winzent::ANN::NeuralNetwork const& network)
{
    out << network.toJSON().toJson();
    return out;
}