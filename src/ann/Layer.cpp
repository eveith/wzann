#include <QObject>
#include <QList>

#include "Neuron.h"
#include "ConstantActivationFunction.h"
#include "Layer.h"

namespace Winzent {
    namespace ANN {
        Layer::Layer(QObject *parent):
                QObject(parent),
                neurons(QList<Neuron*>())
        {
            // Add the initial bias neuron:

            neurons << new Neuron(new ConstantActivationFunction(), this);
        }


        int Layer::size() const
        {
            return neurons.size() - 1;
        }


        bool Layer::contains(const Neuron *neuron) const
        {
            return neurons.contains(const_cast<Neuron*>(neuron));
        }


        Neuron*& Layer::neuronAt(const int &index)
        {
            return neurons[index];
        }


        Neuron*& Layer::operator [](const int &index)
        {
            return neuronAt(index);
        }


        Neuron*& Layer::biasNeuron()
        {
            return neurons.last();
        }


        Layer& Layer::operator <<(Neuron *neuron)
        {
            // Insert the neuron just before the bias neuron.

            neuron->setParent(this);
            neurons.insert(neurons.size() - 1, neuron);

            Q_ASSERT(biasNeuron() == neurons.last());
            return *this;
        }


        Layer* Layer::clone() const
        {
            Layer* layerClone = new Layer();

            foreach (Neuron* n, neurons) {
                Neuron* neuronClone = n->clone();
                neuronClone->setParent(layerClone);
                layerClone->neurons << neuronClone;
            }

            return layerClone;
        }


    } // namespace ANN
} // namespace Winzent
