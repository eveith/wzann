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
            // Add the initial bias neuron:

            m_neurons << new Neuron(new ConstantActivationFunction(), this);
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
            return m_neurons.size() - 1;
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


        Neuron *const &Layer::biasNeuron()
        {
            return m_neurons.last();
        }


        const Neuron *Layer::biasNeuron() const
        {
            return m_neurons.last();
        }


        void Layer::eachNeuron(function<void (const Neuron * const &)> yield)
                const
        {
            std::for_each(m_neurons.begin(), (m_neurons.end() - 1), yield);
        }


        void Layer::eachNeuron(function<void (Neuron * const &)> yield)
        {
            std::for_each(m_neurons.begin(), (m_neurons.end() - 1), yield);
        }


        Layer &Layer::operator <<(Neuron *neuron)
        {
            // Insert the neuron just before the bias neuron.

            neuron->setParent(this);
            m_neurons.insert(m_neurons.size() - 1, neuron);

            Q_ASSERT(biasNeuron() == m_neurons.last());
            return *this;
        }


        Layer *Layer::clone() const
        {
            return new Layer(*this);
        }


    } // namespace ANN
} // namespace Winzent
