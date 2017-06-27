#include <cstddef>
#include <cassert>
#include <functional>

#include <boost/ptr_container/ptr_vector.hpp>

#include "Neuron.h"
#include "Layer.h"


namespace Winzent {
    namespace ANN {


        Layer::Layer(): m_parent(nullptr)
        {
        }


        Layer::~Layer()
        {
        }


        Layer::size_type Layer::size() const
        {
            return m_neurons.size();
        }


        NeuralNetwork* Layer::parent() const
        {
            return m_parent;
        }


        bool Layer::contains(Neuron const& neuron) const
        {
            return (neuron.parent() == this);
        }


        Neuron* Layer::neuronAt(Layer::size_type index) const
        {
            return const_cast<Neuron *>(&(m_neurons.at(index)));
        }


        Neuron& Layer::operator [](Layer::size_type index)
        {
            return m_neurons[index];
        }


        Neuron const& Layer::operator [](Layer::size_type index) const
        {
            return m_neurons[index];
        }


        Vector Layer::activate(Vector const& neuronInputs)
        {
            assert(neuronInputs.size() == size());

            Vector result;
            result.reserve(size());

            auto iit = neuronInputs.begin();
            auto nit = begin();

            for (; iit != neuronInputs.end() && nit != end(); iit++, nit++) {
                result.push_back(nit->activate(*iit));
            }

            return result;
        }


        Layer::size_type Layer::indexOf(Neuron const& neuron) const
        {
           return m_neuronIndexes.at(const_cast<Neuron *>(&neuron));
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


        Layer& Layer::operator <<(Neuron* const& neuron)
        {
            return addNeuron(neuron);
        }


        Layer& Layer::addNeuron(Neuron* const& neuron)
        {
            neuron->m_parent = this;
            m_neurons.push_back(neuron);
            m_neuronIndexes[neuron] = size()-1;

            return *this;
        }


        Layer* Layer::clone() const
        {
            Layer* clonedLayer = new Layer();

            for (auto const& n: m_neurons) {
                clonedLayer->addNeuron(n.clone());
            }

            return clonedLayer;
        }


        bool Layer::operator ==(Layer const& other) const
        {
            auto i1 = begin();
            auto i2 = other.begin();

            for (; i1 != end() && i2 != other.end(); i1++, i2++) {
                if (*i1 != *i2) {
                    return false;
                }
            }

            return i1 == end() && i2 == other.end();
        }


        bool Layer::operator !=(Layer const& other) const
        {
            return !(*this == other);
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
