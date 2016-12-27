#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_


#include <QObject>
#include <QJsonDocument>

#include <JsonSerializable.h>

#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {

        /*!
         * Represents an activation function within a neural net,
         * implementing the strategy pattern. Instances of this class
         * represent the actual, concrete function, like a sigmoid.
         */
        class WINZENTANNSHARED_EXPORT ActivationFunction:
                public QObject,
                public JsonSerializable
        {
            Q_OBJECT

        public:


            /*!
             * \brief Constructs a new activation function
             *  with, optionaly, a steepness.
             *
             * \param[in] steepness The function's steepness
             *
             * \param[in] parent The parent object (for auto-destruction)
             */
            ActivationFunction(const qreal &steepness = 1.0);


            virtual ~ActivationFunction();


            /*!
             * \brief Returns the activation function's steepness.
             *
             * \return The function's steepness
             */
            qreal steepness() const;


            /*!
             * \brief Sets a new steepness value for this activtion function.
             *
             * \return <code>this</code>, for method chaining.
             */
            ActivationFunction &steepness(qreal steepness);


            /*!
             * \brief Applies the activation function to a certain input
             *  value.
             *
             * \param[in] input The value
             *
             * \return f(input)
             */
            virtual qreal calculate(const qreal &input) = 0;


            /*!
             * \brief Applies the derivative of the activation function
             *  to a certain input value.
             *
             * Activation functions in neural networks need to be derivable if
             * you want to use them during training, e.g.,
             * with backpropagation. Although the derivation itself is a
             * purely mathematic task, it has consequences for the
             * activation function's derivation's input --- if the
             * activation function, for example, contains the e
             * function (exp(...)), it will recreate itself on deviation:
             * \f$\frac{d}{dx}e^x = e^x\f$.
             *
             * This means that for some activation functions, the neuron's
             * last input (i.e., \f$net_j\f$) needs to be used, and for
             * some others, its last result (\f$\varphi{}(net_j)\f$). This is
             * the reason for the two parameters <code>input</code> and
             * <code>sum</code>.
             *
             * When implementing an activation function, choose whichever
             * value you need. Both are supplied.
             *
             * \param[in] sum The neuron's last input from the net
             *  (\f$net_j\f$)
             *
             * \param[in] result The neuron's output (\f$\varphi{}(net_j)\f$)
             *
             * \return The derivation applied to the appropriate value.
             *
             * \sa Neuron#lastInput()
             *
             * \sa Neuron#lastResult()
             */
            virtual qreal calculateDerivative(
                    const qreal &sum,
                    const qreal &result) = 0;


            /*!
             * \brief Indicates whether a derivative is available or not.
             *
             * \return <code>true</code>, if a derivative of the
             *  activation function exists; false otherwise.
             */
            virtual bool hasDerivative() const = 0;


            /*!
             * \brief Clones the activation function
             *
             * Clones the current activation function according to
             * the semantics of the concrete implementation. I.e.,
             * if the activation function does not record any state
             * (as most will do), <code>this</code> is returned;
             * otherwise, a new, empty instance with the same
             * parameters (if any) is created.
             *
             * \return A new activation function object.
             */
            virtual ActivationFunction *clone() const = 0;


            /*!
             * \brief Clips a value when it exceeds a certain boundary.
             *
             * \param value The value is subject to clipping
             *
             * \param[in] min The minimum value the `value<` parameter
             *  may have
             *
             * \param[in] max The maximum value the `value` parameter
             *  may have
             *
             * \param A value in the boundaries of min <= value <= max
             */
            qreal clip(const qreal &value, const qreal &min, const qreal &max)
                    const;


            /*!
             * \brief Resets the Activation function as if newly constructed
             */
            virtual void clear() override;


            /*!
             * \brief Serializes the activation function to JSON
             *
             * This method serializes the standard parameters. If a derived
             * activation function object has additional properties beyond
             * steepness, etc., it must override this implementation.
             *
             * \return The JSON representation
             */
            virtual QJsonDocument toJSON() const override;


            /*!
             * \brief Re-initializes the activation function object from JSON
             *
             * This method serializes the standard parameters. If a derived
             * activation function object has additional properties beyond
             * steepness, etc., it must override this implementation.
             *
             * \param[in] json The activation function's JSON representation
             */
            virtual void fromJSON(const QJsonDocument &json) override;


            /*!
             * \brief Checks for equality of two Activation Function object
             *  pointers
             *
             * This method is virtual since derived classes can re-implement
             * and re-use it in order to support equality checking with
             * regards to run-time polymorphism.
             *
             * \param[in] other The other ActivationFunction object
             *
             * \return True if the two are equal, false otherwise.
             */
            virtual bool equals(const ActivationFunction* const& other) const;


        private:


            //! Factor by which the function is to be scaled.
            qreal m_steepness;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* ACTIVATIONFUNCTION_H_ */
