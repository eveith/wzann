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
             * neurons it holds.
             */
            int size() const;


            /*!
             * Returns the neuron at the specified index position, not including
             * the bias neuron.
             *
             * \sa #biasNeuron()
             */
            Neuron*& operator [](const int &index);


            /*!
             * Provides access to the bias neuron.
             */
            Neuron*& biasNeuron();


            /*!
             * Adds a neuron to the layer.
             */
            Layer& operator<<(Neuron *neuron);


            /*!
             * Creates a new, empty layer.
             */
            Layer(QObject *parent = 0);


            /*!
             * Returns a deep copy (clone) of this layer.
             */
            Layer* clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LAYER_H
