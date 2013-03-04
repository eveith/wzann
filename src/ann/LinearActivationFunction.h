/*!
 * \file
 * \author Eric MSP Veith <eveith@gnyu-linux.org>
 */


#ifndef WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
#define WINZENT_ANN_LINEARACTIVATIONFUNCTION_H


#include <QObject>

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
             * Creates a new linear activation function with an optional scaling
             * and transposition.
             *
             * \param steepness Steepness the function: f(x) = steepness * x
             *
             * \param parent The parent object.
             *
             * \sa QObject
             */
            LinearActivationFunction(
                    double steepness = 1.0,
                    QObject *parent = 0);


            virtual double calculate(const double &input);


            virtual double calculateDerivative(const double &);


            virtual bool hasDerivative() const;


            virtual ActivationFunction *clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
