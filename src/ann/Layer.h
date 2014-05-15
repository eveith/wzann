#ifndef WINZENT_ANN_LAYER_H
#define WINZENT_ANN_LAYER_H

#include <QObject>
#include <QList>

#include <functional>

#include "Neuron.h"


using std::function;


namespace Winzent {
    namespace ANN {
        

        /*!
         * Represents a layer in a neural network
         */
        class Layer: public QObject
        {
            Q_OBJECT

        private:


            /*!
             * A list of all neurons the make up this layer.
             */
            QList<Neuron*> m_neurons;


        public:


            /*!
             * Creates a new, empty layer.
             */
            Layer(QObject *parent = 0);


            /*!
             * Copy constructor
             */
            Layer(const Layer &rhs);


            /*!
             * Returns the size of the layer, i.e. the number of
             * neurons it holds.
             */
            int size() const;


            /*!
             * Checks whether a particular neuron is part of this layer.
             */
            bool contains(const Neuron *const &neuron) const;


            /*!
             * Returns the neuron at the specified index position.
             *
             * Retrieves a neuron given its position (index) in the layer.
             */
            Neuron *&neuronAt(const int &index);


            /*!
             * Nonmodifiable, `const` version of the neuronAt command.
             */
            const Neuron *neuronAt(const int &index) const;


            /*!
             * Returns the neuron at the specified index position.
             *
             * Retrieves a neuron given its position (index) in the layer.
             */
            Neuron*& operator [](const int &index);



            /*!
             * \brief Returns the index of a particular neuron
             *
             * \param[in] neuron The neuron
             *
             * \return The index position, or -1 if no item matched.
             */
            int indexOf(const Neuron *const &neuron) const;


            /*!
             * \brief Iterator access to each neuron as a const reference
             *
             * \param yield The lambda called for each neuron.
             */
            void eachNeuron(function<void(const Neuron *const &)> yield) const;


            /*!
             * \brief Iterator access to each neuron as a modifiable reference
             *
             * \param yield The lambda that is called for each neuron.
             */
            void eachNeuron(function<void(Neuron *const &)> yield);


            /*!
             * Adds a neuron to the layer ensuring that the bias neuron always
             * remains the last one.
             */
            Layer& operator<<(Neuron *neuron);


            /*!
             * Returns a deep copy (clone) of this layer.
             */
            Layer *clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LAYER_H
