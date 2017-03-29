#ifndef NEURON_H_
#define NEURON_H_


#include "Serializable.h"
#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {


        class Layer;


        /*!
         * \brief The Neuron class represents a single neuron in a neural
         *  network.
         *
         * This low-level data structure represents a single neuron. It
         * is configured with a certain activation function. It also acts as
         * a cache, storing the last input it received and the last result
         * of its activation.
         *
         * \sa #activationFunction()
         *
         * \sa #activate()
         */
        class Neuron
        {
            friend class Layer;
            friend Neuron* new_from_variant<>(libvariant::Variant const&);


        public:


            /*!
             * \brief Creates a new neuron
             *
             * \sa #activationFunction()
             */
            Neuron();


            /*!
             * \brief Deleted copy constructor
             *
             * No copy constructor exists for the Neuron class. If you need
             * a deep copy of a neuron, call #clone() instead.
             *
             * \sa Neuron#clone()
             */
            Neuron(Neuron const&) = delete;


            Neuron(Neuron&&) = delete;


            virtual ~Neuron();


            /*!
             * Clones this neuron by creating a new one with the
             * same activation function and the same last result.
             *
             * \return A new neuron with the same state as this one
             *
             * \sa ActivationFunction#clone
             * \sa #lastResult
             */
            Neuron* clone() const;


            /*!
             * \brief Returns the parent Layer
             *
             * \return The parent layer, or `nullptr` if the Neuron does not
             *  belong to any Layer
             *
             * \sa Layer#addNeuron()
             */
            Layer* parent() const;


            //! Returns the last network input for this neuron
            double lastInput() const;


            //! Returns the result of the last activation
            double lastResult() const;


            /*!
             * \brief Returns the activation function
             *  this neuron instance uses.
             */
            ActivationFunction activationFunction() const;


            /*!
             * \brief Sets a new activation function
             *
             * \param[in] activationFunction The activation function
             *
             * \return `*this`
             */
            Neuron& activationFunction(ActivationFunction activationFunction);


            /*!
             * Activates the neuron given the input sum of all
             * weights that lead to this neuron. Also stores the
             * result, which can be re-retrievend using #lastResult.
             *
             * \param[in] sum   The input, i.e. the sum of all
             *  weighted inputs of all connections leading to this
             *  neuron.
             *
             * \return The result of the activation
             *
             * \sa #lastResult
             * \sa #m_activationFunction
             */
            double activate(double sum);


            /*!
             * \brief Checks for equality of two Neurons
             *
             * Two neurons are equal if they are configured with the same
             * activation function and have the same cache values.
             *
             * \sa #activationFunction()
             *
             * \sa #lastInput()
             *
             * \sa #lastResult()
             *
             * \return `true` iff `this == &other`
             */
            bool operator ==(const Neuron& other) const;


            //! Checks for inequality of two Neurons
            bool operator !=(const Neuron& other) const;


        private:


            //! Our parent layer
            Layer* m_parent;


            /*!
             * \brief The activation function used for neuron activation
             *
             * \sa #activate()
             */
            ActivationFunction m_activationFunction;


            //! Caches the input that was presented to #activate().
            double m_lastInput;


            //! Caches the result of the last activation
            double m_lastResult;
        };


        template <>
        inline libvariant::Variant to_variant(Neuron const& neuron)
        {
            libvariant::Variant variant;

            variant["lastInput"] = neuron.lastInput();
            variant["lastResult"] = neuron.lastResult();
            variant["activationFunction"] = to_variant(
                    neuron.activationFunction());

            return variant;

        }


        template <>
        inline Neuron* new_from_variant(libvariant::Variant const& variant)
        {
            auto* af = new_from_variant<ActivationFunction>(
                    variant["activationFunction"]);
            auto* n = new Neuron();

            n->m_lastInput = variant["lastInput"].AsDouble();
            n->m_lastResult = variant["lastResult"].AsDouble();
            n->m_activationFunction = ActivationFunction(*af);

            delete af;
            return n;
        }
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURON_H_ */
