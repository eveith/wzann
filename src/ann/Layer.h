#ifndef WINZENT_ANN_LAYER_H
#define WINZENT_ANN_LAYER_H


#include <QJsonDocument>

#include <cstddef>
#include <unordered_map>

#include <boost/ptr_container/ptr_vector.hpp>

#include <JsonSerializable.h>

#include "Vector.h"
#include "Neuron.h"
#include "Winzent-ANN_global.h"


using std::function;


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;


        /*!
         * \brief Represents a layer in a neural network
         */
        class WINZENTANNSHARED_EXPORT Layer: public JsonSerializable
        {
            friend class NeuralNetwork;


        public:


            typedef boost::ptr_vector<Neuron>::iterator iterator;
            typedef boost::ptr_vector<Neuron>::const_iterator const_iterator;
            typedef boost::ptr_vector<Neuron>::size_type size_type;


            /*!
             * \brief Creates a new, empty layer.
             */
            Layer();


            Layer(const Layer &) = delete;
            Layer(Layer &&) = delete;


            virtual ~Layer();



            /*!
             * \brief Returns a deep copy (clone) of this layer.
             *
             * \return The clone
             */
            Layer *clone() const;


            /*!
             * \brief Grants access to the parent neural network
             *
             * \return The parent neural network, or `nullptr` when the layer
             *  isn't contained in any neural network yet
             */
            NeuralNetwork *parent() const;


            /*!
             * \brief Returns the size of the layer, i.e. the number of
             *  neurons it holds.
             */
            size_type size() const;


            /*!
             * \brief Checks whether a particular neuron is part
             *  of this layer.
             */
            bool contains(const Neuron &neuron) const;


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
            Neuron &operator [](const size_type &index);


            /*!
             * \brief Returns the neuron at the specified index position.
             *
             * Requires `index < size()`.
             *
             * \param[in] The index
             *
             * \return The neuron at the given position
             */
            const Neuron &operator [](const size_type &index) const;

            /*!
             * \brief Activates all neurons in this layer.
             *
             * The given vector contains the inputs for the neurons, in order.
             * It must have the same size as the layer.
             *
             * \param[in] neuronInputs The inputs to this layer's neurons.
             *
             * \return The results of the activation.
             */
            Vector activate(const Vector &neuronInputs);


            /*!
             * \brief Returns the index of a particular neuron
             *
             * \param[in] neuron The neuron
             *
             * \return The index position, or -1 if no item matched.
             */
            size_type indexOf(const Neuron &neuron) const;


            /*!
             * \brief Returns a new iterator pointing at the first neuron in
             *  the Layer
             *
             * \return A new iterator
             */
            iterator begin();


            /*!
             * \brief Returns a new const iterator pointing at the first
             *  neuron in the Layer
             *
             * \return A new const iterator
             */
            const_iterator begin() const;


            /*!
             * \brief Returns a new iterator pointing at the non-existent
             *  element after the last Neuron in this Layer
             *
             * \return A new iterator
             */
            iterator end();


            /*!
             * \brief Returns a new const iterator pointing at the
             *  non-existent element after the last Neuron in this Layer
             *
             * \return A new const iterator
             */
            const_iterator end() const;


            /*!
             * \brief Adds a neuron to the layer
             *
             * The takes ownership of the neuron, which will be deleted when
             * the Layer is deleted.
             *
             * \return `this`
             */
            Layer &operator<<(Neuron *const &neuron);


            /*!
             * \brief Adds a neuron to the layer
             *
             * The takes ownership of the neuron, which will be deleted when
             * the Layer is deleted.
             *
             * \return `this`
             */
            Layer &addNeuron(Neuron *const &neuron);


            //! Resets the layer, clearing it.
            virtual void clear() override;


            /*!
             * \brief Serializes the whole Layer to JSON
             *
             * \return The Layer's JSON representation
             */
            virtual QJsonDocument toJSON() const override;


            /*!
             * \brief Deserializes the Layer from JSON
             *
             * \param[in] json The Layer's JSON representation
             */
            virtual void fromJSON(const QJsonDocument &json) override;


            //! Checks for equality
            bool operator ==(const Layer &other) const;


        private:


            //! \brief A list of all neurons the make up this layer.
            boost::ptr_vector<Neuron> m_neurons;


            //! \brief Maps Neurons to their index for faster #indexOf()
            std::unordered_map<Neuron *, size_t> m_neuronIndexes;


            //! The parent network we're contained in.
            NeuralNetwork *m_parent;
        };
    } // namespace ANN
} // namespace Winzent


Winzent::ANN::Layer::iterator begin(
        Winzent::ANN::Layer *const &layer);
Winzent::ANN::Layer::const_iterator begin(
        const Winzent::ANN::Layer *const &layer);
Winzent::ANN::Layer::iterator end(
        Winzent::ANN::Layer *const &layer);
Winzent::ANN::Layer::const_iterator end(
        const Winzent::ANN::Layer *const &layer);


#endif // WINZENT_ANN_LAYER_H
