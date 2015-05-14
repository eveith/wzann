#include <QObject>

#include <cstddef>
#include <functional>

#include <boost/ptr_container/ptr_vector.hpp>

#include "Neuron.h"
#include "Layer.h"


namespace Winzent {
    namespace ANN {
        Layer::Layer(QObject *parent): QObject(parent)
        {
        }


        Layer::Layer(const Layer &rhs): QObject(rhs.parent())
        {
            for (const auto &n: rhs.m_neurons) {
                addNeuron(n.clone());
            }
        }

        size_t Layer::size() const
        {
            return m_neurons.size();
        }


        bool Layer::contains(const Neuron *const &neuron) const
        {
            return (neuron->parent() == this);
        }


        Neuron *Layer::neuronAt(const size_t &index) const
        {
            return const_cast<Neuron *>(&(m_neurons.at(index)));
        }


        Neuron *Layer::operator [](const size_t &index)
        {
            return &(m_neurons[index]);
        }


        size_t Layer::indexOf(const Neuron *const &neuron) const
        {
            size_t index = -1;

            for (size_t i = 0; i != m_neurons.size(); ++i) {
                if (&(m_neurons.at(i)) == neuron) {
                    index = i;
                    break;
                }
            }

            return index;
        }



        void Layer::eachNeuron(function<void (const Neuron * const &)> yield)
                const
        {
            for (const Neuron &n: m_neurons) {
                yield(&n);
            }
        }


        void Layer::eachNeuron(function<void (Neuron *const &)> yield)
        {
            for (Neuron &n: m_neurons) {
                yield(&n);
            }
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

            return *this;
        }


        Layer *Layer::clone() const
        {
            return new Layer(*this);
        }
    } // namespace ANN
} // namespace Winzent


Winzent::ANN::Layer::iterator begin(Winzent::ANN::Layer *&layer)
{
    return layer->begin();
}


Winzent::ANN::Layer::const_iterator begin(const Winzent::ANN::Layer *&layer)
{
    return layer->begin();
}


Winzent::ANN::Layer::iterator end(Winzent::ANN::Layer *&layer)
{
    return layer->end();
}


Winzent::ANN::Layer::const_iterator end(const Winzent::ANN::Layer *&layer)
{
    return layer->end();
}
