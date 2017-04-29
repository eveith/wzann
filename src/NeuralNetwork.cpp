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

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"
#include "NoConnectionException.h"
#include "UnknownNeuronException.h"
#include "LayerSizeMismatchException.h"

#include "NeuralNetwork.h"
#include "WzannGlobal.h"


using std::function;
using log4cxx::LogManager;
using boost::make_iterator_range;


namespace Winzent {
    namespace ANN {

        const char NeuralNetwork::VERSION[] = "1.0";


        NeuralNetwork::NeuralNetwork(): m_biasNeuron(new Neuron())
        {
            m_biasNeuron->activationFunction(ActivationFunction::Identity);
        }


        NeuralNetwork::NeuralNetwork(NeuralNetwork const& rhs):
                m_biasNeuron(rhs.m_biasNeuron->clone())
        {
            // Clone layers:

            for (auto const& l: rhs.m_layers) {
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

                    assert(srcLayerIndex >= 0 && srcNeuronIndex >= 0);

                    connections = rhs.connectionsFrom(*foreignNeuron);
                }

                assert(connections.second-connections.first > 0);

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

                    assert(dstLayerIndex >= 0 && dstNeuronIndex >= 0);

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
        }


        NeuralNetwork *NeuralNetwork::clone() const
        {
            return new NeuralNetwork(*this);
        }


        Neuron const& NeuralNetwork::biasNeuron() const
        {
            return *m_biasNeuron;
        }


        Neuron& NeuralNetwork::biasNeuron()
        {
            return *m_biasNeuron;
        }


        bool NeuralNetwork::contains(Neuron const& neuron) const
        {
            if (&(biasNeuron()) == &neuron) {
                return true;
            }

            return std::any_of(m_layers.begin(), m_layers.end(), [&neuron](
                    Layer const& layer) {
                return layer.contains(neuron);
            });
        }


        bool NeuralNetwork::connectionExists(
                Neuron const& from,
                Neuron const& to)
                const
        {
            auto it = m_connectionSources.find(const_cast<Neuron*>(&from));
            return it != m_connectionSources.end() && std::any_of(
                    it->second.begin(),
                    it->second.end(),
                    [&to](Connection const* const& c) {
                return &(c->destination()) == &to;
            });
        }


        Connection* NeuralNetwork::connection(
                Neuron const& from,
                Neuron const& to)
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

            throw NoConnectionException(from, to);
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
                Neuron const& from,
                Neuron const& to)
        {
            if (&from != &biasNeuron() && ! contains(from)) {
                throw UnknownNeuronException(from);
            }

            if (! contains(to)) {
                throw UnknownNeuronException(to);
            }

            Neuron& src = const_cast<Neuron&>(from),
                    &dst = const_cast<Neuron&>(to);

            auto* connection = new Connection(src, dst, 0.0);

            m_connections.push_back(connection);
            m_connectionSources[&src].push_back(connection);
            m_connectionDestinations[&dst].push_back(connection);

            return *connection;
        }


        void NeuralNetwork::disconnectNeurons(
                Neuron const& from,
                Neuron const& to)
        {
            auto connection = std::find_if(
                    m_connections.begin(),
                    m_connections.end(),
                    [&from, &to](Connection const& c) {
                return &(c.source()) == &from
                        && &(c.destination()) == &to;
            });

            if (connection == m_connections.end()) {
                throw NoConnectionException(from, to);
            }

            auto &sources = m_connectionSources[const_cast<Neuron *>(&from)];
            std::remove(
                    sources.begin(),
                    sources.end(),
                    &(*connection));

            auto &destinations = m_connectionDestinations.at(
                    const_cast<Neuron *>(&to));
            std::remove(
                    destinations.begin(),
                    destinations.end(),
                    &(*connection));

            m_connections.erase(connection);
            delete &(*connection);
        }


        NeuralNetwork& NeuralNetwork::operator <<(Layer* layer)
        {
            layer->m_parent = this;
            m_layers.push_back(layer);
            return *this;
        }


        Layer* NeuralNetwork::layerAt(size_type index) const
        {
            return &(const_cast<NeuralNetwork*>(this)->m_layers.at(index));
        }


        Layer const& NeuralNetwork::operator [](size_type index) const
        {
            return m_layers[index];
        }


        Layer& NeuralNetwork::operator [](size_type index)
        {
            return m_layers[index];
        }


        Layer& NeuralNetwork::inputLayer()
        {
            return m_layers.front();
        }


        Layer& NeuralNetwork::outputLayer()
        {
            return m_layers.back();
        }


        NeuralNetwork::LayerConstRange NeuralNetwork::layers() const
        {
            return std::make_pair(m_layers.begin(), m_layers.end());
        }


        NeuralNetwork::LayerRange NeuralNetwork::layers()
        {
            return std::make_pair(m_layers.begin(), m_layers.end());
        }


        NeuralNetwork &NeuralNetwork::configure(
                NeuralNetworkPattern const& pattern)
        {
            m_pattern.reset(pattern.clone());
            m_pattern->configureNetwork(*this);
            return *this;
        }


        Vector NeuralNetwork::calculateLayerTransition(
                Layer const& from,
                Layer const& to,
                Vector const& input)
        {
#ifdef WZANN_DEBUG
            auto fromLayerSize = from.size();
            if (static_cast<Layer::size_type>(input.size()) != fromLayerSize){
                throw LayerSizeMismatchException(fromLayerSize, input.size());
            }
#endif

            Vector output;
            auto toLayerSize = to.size();
            output.resize(toLayerSize, 0.0);

            for (Layer::size_type t = 0; t != toLayerSize; ++t) {
                const Neuron &toNeuron = to[t];
                auto connections = connectionsTo(toNeuron);

                for (const auto &c: make_iterator_range(connections)) {
                    assert(&(c->destination()) == &toNeuron);

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
                Vector const& input)
        {
            Vector biasedInput(input.size());

            auto it = layer.begin();
            for (Vector::size_type i = 0; i != input.size(); ++i, it++) {
                auto &neuron = *it;
                biasedInput[i] = input[i];

                if (! connectionExists(biasNeuron(), neuron)) {
                    continue;
                }

                auto const& c = connection(biasNeuron(), neuron);
                biasedInput[i] += biasNeuron().activate(1.0) * c->weight();
            }

            assert(it == layer.end());
            return layer.activate(biasedInput);
        }


        Vector NeuralNetwork::calculate(Vector const& input)
        {
            if (static_cast<Layer::size_type>(input.size())
                    != m_layers.front().size()) {
                throw LayerSizeMismatchException(
                        m_layers.front().size(),
                        input.size());
            }

            return m_pattern->calculate(*this, input);
        }


        bool NeuralNetwork::operator ==(const NeuralNetwork &other) const
        {
            bool equal = true;

            equal &= size() == other.size();
            equal &= biasNeuron() == other.biasNeuron();
            equal &= ((m_pattern == nullptr && other.m_pattern == nullptr)
                    || (m_pattern != nullptr && other.m_pattern != nullptr
                        && *m_pattern == *(other.m_pattern)));
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
                        if (**it1 != **it2) {
                            return false;
                        }
                    }
                }
            }


            return equal;
        }


        bool NeuralNetwork::operator !=(NeuralNetwork const& other) const
        {
            return !(*this == other);
        }
    }
}
