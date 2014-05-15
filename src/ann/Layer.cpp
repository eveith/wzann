#include <QObject>
#include <QList>

#include "Neuron.h"
#include "ConstantActivationFunction.h"
#include "Layer.h"

namespace Winzent {
    namespace ANN {
        Layer::Layer(QObject *parent):
                QObject(parent)
        {
        }


        Layer::Layer(const Layer &rhs):
                QObject(rhs.parent()),
                m_neurons(QList<Neuron *>())
        {
            foreach (Neuron *n, rhs.m_neurons) {
                Neuron *neuronClone = n->clone();
                neuronClone->setParent(this);
                m_neurons << neuronClone;
            }
        }


        int Layer::size() const
        {
            return m_neurons.size();
        }


        bool Layer::contains(const Neuron *const &neuron) const
        {
            return m_neurons.contains(const_cast<Neuron*>(neuron));
        }


        Neuron *&Layer::neuronAt(const int &index)
        {
            return m_neurons[index];
        }


        const Neuron *Layer::neuronAt(const int &index) const
        {
            return m_neurons.at(index);
        }


        Neuron *&Layer::operator [](const int &index)
        {
            return neuronAt(index);
        }


        int Layer::indexOf(const Neuron *const &neuron) const
        {
            return m_neurons.indexOf(const_cast<Neuron *>(neuron));
        }



        void Layer::eachNeuron(function<void (const Neuron * const &)> yield)
                const
        {
            std::for_each(m_neurons.begin(), m_neurons.end(), yield);
        }


        void Layer::eachNeuron(function<void (Neuron *const &)> yield)
        {
            std::for_each(m_neurons.begin(), m_neurons.end(), yield);
        }


        Layer &Layer::operator <<(Neuron *neuron)
        {
            neuron->setParent(this);
            m_neurons.append(neuron);

            return *this;
        }


        Layer *Layer::clone() const
        {
            return new Layer(*this);
        }
    } // namespace ANN
} // namespace Winzent
