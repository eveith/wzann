#include <QObject>

#include <QList>
#include <QVector>

#include <QByteArray>
#include <QTextStream>

#include <QJsonValue>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <memory>
#include <cstddef>
#include <functional>

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


namespace Winzent {
    namespace ANN {

        const char NeuralNetwork::VERSION[] = "1.0";


        log4cxx::LoggerPtr NeuralNetwork::logger =
                log4cxx::LogManager::getLogger("Winzent.ANN.NeuralNetwork");


        NeuralNetwork::NeuralNetwork():
                m_biasNeuron(new Neuron(new ConstantActivationFunction()))
        {
            Q_ASSERT(m_connectionSources.size()
                     == m_connectionDestinations.size());
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork &rhs):
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
                for (Neuron& n: *layer) {
                    QList<Connection *> connections =
                            neuronConnectionsFrom(&n);
                    std::for_each(
                            connections.begin(),
                            connections.end(),
                            yield);
                }
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
                for (const Neuron& neuron: *layer) {
                    QList<Connection *> connections =
                            neuronConnectionsFrom(&neuron);
                    std::for_each(
                            connections.begin(),
                            connections.end(),
                            yield);
                }
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


        Layer &NeuralNetwork::inputLayer()
        {
            return m_layers.front();
        }


        Layer &NeuralNetwork::outputLayer()
        {
            return m_layers.back();
        }


        NeuralNetwork::LayerIterator NeuralNetwork::begin()
        {
            return m_layers.begin();
        }


        NeuralNetwork::LayerConstIterator NeuralNetwork::begin() const
        {
            return m_layers.begin();
        }


        NeuralNetwork::LayerIterator NeuralNetwork::end()
        {
            return m_layers.end();
        }


        NeuralNetwork::LayerConstIterator NeuralNetwork::end() const
        {
            return m_layers.end();
        }


        NeuralNetwork &NeuralNetwork::configure(
                const NeuralNetworkPattern &pattern)
        {
            m_pattern.reset(pattern.clone());
            m_pattern->configureNetwork(this);
            return *this;
        }


        Vector NeuralNetwork::calculateLayer(
                Layer *const &layer,
                const Vector &input)
                    throw(LayerSizeMismatchException)
        {
#ifdef QT_DEBUG
            if (layer->size() != input.size()) {
                throw LayerSizeMismatchException(input.size(), layer->size());
            }
#endif

            Vector output;
            output.reserve(layer->size());

            for (int i = 0; i != input.size(); ++i) {
                qreal sum = input.at(i);
                Neuron *neuron = layer->neuronAt(i);

                // Add bias neuron. We ignore the bias neuron in the input layer
                // even when it's there; it does not make sense to include the
                // bias neuron in the input layer since its output would be
                // overwritten by the input anyways.

                if (&(inputLayer()) != layer
                        && neuronConnectionExists(biasNeuron(), neuron)) {
                    sum += neuronConnection(biasNeuron(), neuron)->weight()
                            * biasNeuron()->activate(1.0);
                }

                output << neuron->activate(sum);
            }

            return (output);
        }


        Vector NeuralNetwork::calculateLayer(
                const int &layerIndex,
                const Vector &input)
                    throw(LayerSizeMismatchException)
        {
            return this->calculateLayer(layerAt(layerIndex), input);
        }


        Vector NeuralNetwork::calculateLayerTransition(
                const int &from,
                const int &to,
                const Vector &input)
                    throw(LayerSizeMismatchException)
        {
            Layer *fromLayer    = layerAt(from);
            Layer *toLayer      = layerAt(to);
            size_t fromLayerSize= fromLayer->size();
            size_t toLayerSize  = toLayer->size();

#ifdef QT_DEBUG
            if (input.size() != fromLayerSize) {
                throw LayerSizeMismatchException(input.size(), fromLayerSize);
            }
#endif

            Vector output;
            output.fill(0.0, toLayerSize);
            Q_ASSERT(output.size() == toLayerSize);

            for (size_t i = 0; i != fromLayerSize; ++i) {
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


        Vector NeuralNetwork::calculate(const Vector &input)
                throw(LayerSizeMismatchException)
        {
            if (input.size() != m_layers.front().size()) {
                throw LayerSizeMismatchException(
                        m_layers.front().size(),
                        input.size());
            }

            return m_pattern->calculate(this, input);
        }


        void NeuralNetwork::clear()
        {
            for (auto const& k: m_connectionSources.keys()) {
                for (auto& c: m_connectionSources[k]) {
                    delete c;
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

                Connection *connection = nullptr;

                if (c["srcNeuron"] == "BIAS") {
                    connection = neuronConnection(
                            m_biasNeuron,
                            layerAt(c["dstLayer"].toInt())
                                ->neuronAt(c["dstNeuron"].toInt()));
                } else {
                    connection = connectNeurons(
                            layerAt(c["srcLayer"].toInt())->neuronAt(
                                c["srcNeuron"].toInt()),
                            layerAt(c["dstLayer"].toInt())->neuronAt(
                                c["dstNeuron"].toInt()));
                }

                connection->weight(c["weight"].toDouble());
                connection->fixedWeight(c["fixedWeight"].toBool());
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
            for (Neuron* const& n: m_connectionSources.keys()) {
                for (Connection* const& c: m_connectionSources.value(n)) {
                    int srcLayer = -1,
                            dstLayer = -1,
                            srcNeuron = -1,
                            dstNeuron = -1;
                    QJsonObject connection;

                    for (size_t i = 0; i != m_layers.size(); ++i) {
                        for (size_t j = 0; j != m_layers.at(i).size(); ++j) {
                            Neuron* const& n = layerAt(i)->neuronAt(j);

                            if (n == c->source()) {
                                srcLayer = i;
                                srcNeuron = j;
                            } else if (n == c->destination()) {
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
            equal &= *m_biasNeuron == *(other.m_biasNeuron);
            equal &= (m_pattern == nullptr && other.m_pattern == nullptr
                    || m_pattern != nullptr && other.m_pattern != nullptr
                        && m_pattern->equals(other.m_pattern.get()));
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

            for (size_t i = 0; i != size(); ++i) {
                const auto& layer = layerAt(i);

                for (size_t j = 0; j != layer->size(); ++j) {
                    const auto& ourConnections = neuronConnectionsFrom(
                            layer->neuronAt(j));
                    const auto& otherConnections =
                            other.neuronConnectionsFrom(
                                other.layerAt(i)->neuronAt(j));

                    for (auto it1 = ourConnections.constBegin(),
                                it2 = otherConnections.constBegin();
                            it1 != ourConnections.constEnd()
                                && it2 != otherConnections.constEnd();
                            it1++, it2++) {
                        if (**it1 != **it2) {
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
