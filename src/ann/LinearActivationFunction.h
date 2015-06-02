/*!
 * \file
 * \author Eric MSP Veith <eveith@gnyu-linux.org>
 */


#ifndef WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
#define WINZENT_ANN_LINEARACTIVATIONFUNCTION_H


#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {


        /*!
         * The linear activation function resembles a simple linear function.
         *
         * Basically, this function is the equivalent of f(x) = x, with an
         * optional scaling and transposition applied, so that is can become
         * f(x) = ax + c.
         *
         * The typical use case for this activation function is the input layer.
         */
        class LinearActivationFunction: public ActivationFunction
        {
        public:


            /*!
             * \brief Creates a new linear activation function with
             *  an optional scaling and transposition.
             *
             * \param steepness Steepness the function: f(x) = steepness * x
             */
            LinearActivationFunction(double steepness = 1.0);


            /*!
             * \return `input * steepness()`
             */
            virtual double calculate(const double &input) override;


            /*!
             * \return ActivationFunction#steepness()
             */
            virtual double calculateDerivative(const double &, const double &)
                    override;


            /*!
             * \return `true`
             */
            virtual bool hasDerivative() const override;


            /*!
             * \brief Clones the activation function
             *
             * \return A clone of this object
             */
            virtual ActivationFunction *clone() const override;
        };

    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
