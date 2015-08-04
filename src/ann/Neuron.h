#ifndef NEURON_H_
#define NEURON_H_


#include <memory>
#include <QJsonDocument>
#include <JsonSerializable.h>

#include "Winzent-ANN_global.h"


using std::shared_ptr;


namespace Winzent {
    namespace ANN {


        class Layer;
        class NeuralNetwork;
        class ActivationFunction;


        class WINZENTANNSHARED_EXPORT Neuron: public JsonSerializable
        {
            friend class Layer;


        public:


            /*!
             * \brief Creates a new neuron with a specific activation
             * function
             *
             * \param[in] activationFunction The activation function
             *  that is used to calculate the neuron's activation.
             *  The Neuron object takes ownership of the activation
             *  function object. If a shared activation function is desiered,
             *  use the setter that takes a std::shared_ptr.
             *
             * \sa #activate
             */
            Neuron(ActivationFunction *const &activationFunction);


            /*!
             * \brief Constructs a new Neuron with a shared activation
             *  function
             *
             * \param[in] activationFunction The activation function for this
             *  neuron that is shared with other neurons
             */
            Neuron(shared_ptr<ActivationFunction> &activationFunction);


            /*!
             * \brief Deleted copy constructor
             *
             * No copy constructor exists for the Neuron class. If you need
             * a deep copy of a neuron, call #clone() instead.
             *
             * \sa Neuron#clone()
             */
            Neuron(const Neuron &) = delete;


            Neuron(Neuron &&) = delete;


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
            Neuron *clone() const;


            /*!
             * \brief Returns the parent Layer
             *
             * \return The parent layer, or `nullptr` if the Neuron does not
             *  belong to any Layer
             *
             * \sa Layer#addNeuron()
             */
            Layer *parent() const;


            //! Returns the last network input for this neuron
            qreal lastInput() const;



            //! Returns the result of the last activation
            qreal lastResult() const;


            /*!
             * \brief Returns the activation function
             *  this neuron instance uses.
             */
            ActivationFunction *activationFunction() const;


            /*!
             * \brief Sets a new activation function
             *
             * \param[in] activationFunction The activation function
             *
             * \return `*this`
             */
            Neuron &activationFunction(
                    ActivationFunction *const &activationFunction);

            /*!
             * \brief Sets a new activation function
             *
             * Invoking this method explicitly allows to share an
             * ActivationFunction object with other neurons.
             *
             * \param[in] activationFunction The activation function
             *
             * \return `*this`
             */
            Neuron &activationFunction(
                    shared_ptr<ActivationFunction> &activationFunction);


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
            qreal activate(const qreal &sum);


            //! Resets the neuron to a pristine state
            virtual void clear() override;


            /*!
             * \brief Serializes the Neuron object to JSON
             *
             * \return The JSON representation of the neuron
             */
            virtual QJsonDocument toJSON() const override;


            /*!
             * \brief Deserializes the Neuron object from JSON
             *
             * \param[in] json The JSON representation of the Neuron
             */
            virtual void fromJSON(const QJsonDocument& json) override;


            //! Checks for equality of two Neurons
            bool operator ==(const Neuron& other) const;


            //! Checks for inequality of two Neurons
            bool operator !=(const Neuron& other) const;


        private:


            //! Our parent layer
            Layer *m_parent;


            /*!
             * \brief The activation function used for neuron activation
             *
             * \sa #activate()
             */
            std::shared_ptr<ActivationFunction> m_activationFunction;


            //! Caches the input that was presented to #activate().
            qreal m_lastInput;


            //! Caches the result of the last activation
            qreal m_lastResult;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURON_H_ */
