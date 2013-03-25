#ifndef WINZENT_ANN_LAYER_H
#define WINZENT_ANN_LAYER_H

#include <QObject>
#include <QList>

#include "Neuron.h"


namespace Winzent {
    namespace ANN {
        

        /*!
         * Represents a layer in a neural network
         */
        class Layer : public QObject
        {
            Q_OBJECT

        public:


            /*!
             * A list of all neurons the make up this layer.
             */
            QList<Neuron*> neurons;


            /*!
             * Returns the size of the layer, i.e. the number of
             * neurons it holds. Excludes the bias neuron.
             */
            int size() const;


            /*!
             * Checks whether a particular neuron is part of this layer.
             */
            bool contains(const Neuron *neuron) const;


            /*!
             * Returns the neuron at the specified index position.
             *
             * Retrieves a neuron given its position (index) in the layer. One
             * can access the bias neuron, too, using
             * <code>layer[layer.size()]</code>.
             *
             * \sa #biasNeuron()
             */
            Neuron*& neuronAt(const int &index);


            /*!
             * Nonmodifiable, `const` version of the neuronAt command.
             */
            const Neuron* neuronAt(const int &index) const;


            /*!
             * Returns the neuron at the specified index position.
             *
             * Retrieves a neuron given its position (index) in the layer. One
             * can access the bias neuron, too, using
             * <code>layer[layer.size()]</code>.
             *
             * \sa #biasNeuron()
             */
            Neuron*& operator [](const int &index);


            /*!
             * Provides access to the bias neuron.
             */
            Neuron*& biasNeuron();


            /*!
             * Adds a neuron to the layer ensuring that the bias neuron always
             * remains the last one.
             */
            Layer& operator<<(Neuron *neuron);


            /*!
             * Creates a new, empty layer.
             */
            Layer(QObject *parent = 0);


            /*!
             * Copy constructor
             */
            Layer(const Layer &rhs);


            /*!
             * Returns a deep copy (clone) of this layer.
             */
            Layer* clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LAYER_H
