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
             */
            virtual double calculateDerivative(const double& input) = 0;


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
