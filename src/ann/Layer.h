#ifndef WINZENT_ANN_LAYER_H
#define WINZENT_ANN_LAYER_H

#include <QObject>

#include <cstddef>
#include <functional>

#include <boost/ptr_container/ptr_vector.hpp>


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
             * \brief A list of all neurons the make up this layer.
             */
            boost::ptr_vector<Neuron> m_neurons;


        public:


            /*!
             * \brief Creates a new, empty layer.
             */
            Layer(QObject *parent = 0);


            /*!
             * \brief Copy constructor
             */
            Layer(const Layer &rhs);


            /*!
             * \brief Returns the size of the layer, i.e. the number of
             *  neurons it holds.
             */
            size_t size() const;


            /*!
             * \brief Checks whether a particular neuron is part
             *  of this layer.
             */
            bool contains(const Neuron *const &neuron) const;


            /*!
             * \brief Returns the neuron at the specified index position.
             *
             * Requires `index < size()`.
             *
             * \param[in] The index
             *
             * \return The neuron at the given position
             */
            Neuron *neuronAt(const size_t &index) const;


            /*!
             * \brief Returns the neuron at the specified index position.
             *
             * Requires `index < size()`.
             *
             * \param[in] The index
             *
             * \return The neuron at the given position
             */
            Neuron *operator [](const size_t &index);



            /*!
             * \brief Returns the index of a particular neuron
             *
             * \param[in] neuron The neuron
             *
             * \return The index position, or -1 if no item matched.
             */
            size_t indexOf(const Neuron *const &neuron) const;


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
             * \brief Adds a neuron to the layer
             *
             * The takes ownership of the neuron, which will be deleted when
             * the Layer is deleted.
             *
             * \return `this`
             */
            Layer& operator<<(Neuron *const &neuron);


            /*!
             * \brief Adds a neuron to the layer
             *
             * The takes ownership of the neuron, which will be deleted when
             * the Layer is deleted.
             *
             * \return `this`
             */
            Layer &addNeuron(Neuron *const &neuron);


            /*!
             * \brief Returns a deep copy (clone) of this layer.
             *
             * \return The clone
             */
            Layer *clone() const;
        };

    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LAYER_H
