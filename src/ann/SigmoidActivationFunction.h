/*
 * SigmoidActivationFunction.h
 *
 *  Created on: 04.11.2012
 *      Author: eveith
 */

#ifndef SIGMOIDACTIVATIONFUNCTION_H_
#define SIGMOIDACTIVATIONFUNCTION_H_


#include <cmath>

#include "ActivationFunction.h"


namespace Winzent
{
    namespace ANN
    {

        class SigmoidActivationFunction: public ActivationFunction
        {
            Q_OBJECT


        public:


            SigmoidActivationFunction(
                    double steepness = 1.0,
                    QObject *parent = 0);


            /*!
             * Calculates output using the sigmoid function.
             */
            virtual double calculate(const double &input);


            /*!
             * Calculates the output of the derivative of the sigmoid
             * function.
             *
             * \sa ActivationFunction#calculateDerivative
             */
            virtual double calculateDerivative(
                    const double &,
                    const double &result);


            /*!
             * Indicates that this function has a derivative.
             *
             * \return `true`
             */
            virtual bool hasDerivative() const {
                return true;
            }


            /*!
             * Clones this activation function and returns a new
             * instance.
             *
             * \return A new instance of this class
             */
            virtual ActivationFunction* clone() const;
        };

    } /* namespace ANN */
} /* namespace Winzent */

#endif /* SIGMOIDACTIVATIONFUNCTION_H_ */
