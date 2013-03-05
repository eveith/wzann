/*
 * ActivationFunction.h
 *
 *  Created on: 16.10.2012
 *      Author: eveith
 */

#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_


#include <QObject>


namespace Winzent
{
    namespace ANN
    {

        /*!
         * Represents an activation function within a neural net,
         * implementing the strategy pattern. Instances of this class
         * represent the actual, concrete function, like a sigmoid.
         */
        class ActivationFunction: public QObject
        {
            Q_OBJECT


        private:


            /*!
             * Factor by which the function is to be scaled.
             */
            double m_steepness;


        public:


            /*!
             * Constructs a new activation function with an optional steepness.
             *
             * \param steepness The function's steepness
             *
             * \param parent The parent object (for auto-destruction)
             *
             * \sa QObject
             */
            ActivationFunction(double steepness = 1.0, QObject *parent = 0);


            /*!
             * Returns the activation function's steepness.
             */
            double steepness() const;


            /*!
             * Sets a new steepness value for this activtion function.
             *
             * \return <code>this</code>, for method chaining.
             */
            ActivationFunction* steepness(double steepness);


            /*!
             * Applies the activation function to a certain input
             * value.
             */
            virtual double calculate(const double& input) = 0;


            /*!
             * Applies the derivative of the activation function
             * to a certain input value.
             *
             * Activation functions in neural networks need to be deriveable if
             * you want to use them during training, e.g. with backpropagation.
             * Although the derivation itself is a purely mathematic task, it
             * has consequences for the activation function's derivation's input
             * --- if the activation function, for example, contains the e
             * function (exp(...)), it will recreate itself on deviation:
             * \f$\frac{d}{dx}e^x = e^x\f$.
             *
             * This means that for some activation functions, the neuron's last
             * input (i.e., \f$net_j\f$) needs to be used, and for some others,
             * its last result (\f$\varphi{}(net_j)\f$). This is the reason for
             * the two parameters <code>input</code> and <code>sum</code>.
             *
             * When implementing an activation function, choose whichever value
             * you need. Both are supplied.
             *
             * \param sum The neuron's last input from the net (\f$net_j\f$)
             *
             * \param result The neuron's output (\f$\varphi{}(net_j)\f$)
             *
             * \return The derivation applied to the appropriate value.
             *
             * \sa Neuron#lastInput
             *
             * \sa Neuron#lastResult
             */
            virtual double calculateDerivative(
                    const double &sum,
                    const double &result) = 0;


            /*!
             * Indicates whether a derivative is available or not.
             *
             * \return <code>true</code>, if a derivative of the
             *  activation function exists; false otherwise.
             */
            virtual bool hasDerivative() const = 0;


            /*!
             * Clones the current activation function according to
             * the semantics of the concrete implementation. I.e.,
             * if the activation function does not record any state
             * (as most will do), <code>this</code> is returned;
             * otherwise, a new, empty instance with the same
             * parameters (if any) is created.
             *
             * \return A new activation function object.
             */
            virtual ActivationFunction* clone() const = 0;


            /*!
             * Clips a value when it exceeds a certain boundary.
             *
             * \param value The value is subject to clipping
             *
             * \param[in] min The minimum value the <code>value</code> parameter
             *  may have
             *
             * \param[in] max The maximum value the <code>value</code> parameter
             *  may have
             *
             * \param A value in the boundaries of min <= value <= max
             */
            double clip(double value, const double &min, const double &max)
                    const;
        };

    } /* namespace ANN */
} /* namespace Winzent */

#endif /* ACTIVATIONFUNCTION_H_ */
