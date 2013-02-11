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


        Neuron*& Layer::operator [](const int &index)
        {
            return neurons[index];
        }


        Neuron*& Layer::biasNeuron()
        {
            return neurons.last();
        }


        Layer& Layer::operator <<(Neuron *neuron)
        {
            // Insert the neuron just before the bias neuron.

            neurons.insert(neurons.size() - 1, neuron);
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
