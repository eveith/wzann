#include <cstddef>
#include <functional>

#include <QMap>
#include <QVector>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <boost/ptr_container/ptr_vector.hpp>

#include <JsonSerializable.h>

#include "Neuron.h"
#include "Layer.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        Layer::Layer(): JsonSerializable(), m_parent(nullptr)
        {
        }


        Layer::~Layer()
        {
        }


        Layer::size_type Layer::size() const
        {
            return m_neurons.size();
        }


        NeuralNetwork *Layer::parent() const
        {
            return m_parent;
        }


        bool Layer::contains(const Neuron *const &neuron) const
        {
            return (neuron->parent() == this);
        }


        Neuron *Layer::neuronAt(const size_t &index) const
        {
            return const_cast<Neuron *>(&(m_neurons.at(index)));
        }


        Neuron &Layer::operator [](const Layer::size_type &index)
        {
            return m_neurons[index];
        }


        Vector Layer::activate(const Vector &neuronInputs)
        {
            Q_ASSERT(static_cast<Layer::size_type>(neuronInputs.size())
                     == size());

            Vector result;
            result.reserve(size());

            auto iit = neuronInputs.constBegin();
            auto nit = begin();

            for (; iit != neuronInputs.constEnd() && nit != end();
                    iit++, nit++) {
                result.push_back(nit->activate(*iit));
            }

            return result;
        }


        Layer::size_type Layer::indexOf(const Neuron *const &neuron) const
        {
           return m_neuronIndexes.value(const_cast<Neuron* const&>(neuron));
        }


        Layer::iterator Layer::begin()
        {
            return m_neurons.begin();
        }


        Layer::const_iterator Layer::begin() const
        {
            return m_neurons.begin();
        }


        Layer::iterator Layer::end()
        {
            return m_neurons.end();
        }


        Layer::const_iterator Layer::end() const
        {
            return m_neurons.end();
        }


        Layer &Layer::operator <<(Neuron *const &neuron)
        {
            return addNeuron(neuron);
        }


        Layer &Layer::addNeuron(Neuron *const &neuron)
        {
            neuron->m_parent = this;
            m_neurons.push_back(neuron);
            m_neuronIndexes[neuron] = size()-1;

            return *this;
        }


        Layer *Layer::clone() const
        {
            Layer *clonedLayer = new Layer();

            for (const auto &n: m_neurons) {
                clonedLayer->addNeuron(n.clone());
            }

            return clonedLayer;
        }


        void Layer::clear()
        {
            m_neurons.clear();
            m_neuronIndexes.clear();
        }


        QJsonDocument Layer::toJSON() const
        {
            QJsonArray a;

            for (const Neuron &n: m_neurons) {
                a.push_back(n.toJSON().object());
            }

            return QJsonDocument(a);
        }


        void Layer::fromJSON(const QJsonDocument &json)
        {
            clear();
            QJsonArray a = json.array();

            for (const auto &i: a) {
                Neuron *n = new Neuron(nullptr);
                n->fromJSON(QJsonDocument(i.toObject()));
                addNeuron(n);
            }
        }


        bool Layer::operator ==(const Layer &other) const
        {
            auto i1 = begin();
            auto i2 = other.begin();

            for (; i1 != end() && i2 != other.end(); i1++, i2++) {
                if (! (*i1 == *i2)) {
                    return false;
                }
            }

            return i1 == end() && i2 == other.end();
        }
    } // namespace ANN
} // namespace Winzent


Winzent::ANN::Layer::iterator begin(
        Winzent::ANN::Layer *const &layer)
{
    return layer->begin();
}


Winzent::ANN::Layer::const_iterator begin(
        const Winzent::ANN::Layer *const &layer)
{
    return layer->begin();
}


Winzent::ANN::Layer::iterator end(
        Winzent::ANN::Layer *const &layer)
{
    return layer->end();
}


Winzent::ANN::Layer::const_iterator end(
        const Winzent::ANN::Layer *const &layer)
{
    return layer->end();
}
