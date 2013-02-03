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
                    double scale = 1.0,
                    double transposition = 0.0,
                    QObject *parent = 0);


            /*!
             * Calculates output using the sigmoid function.
             */
            virtual double calculate(const double& input);


            /*!
             * Calculates the output of the derivative of the sigmoid
             * function.
             */
            virtual double calculateDerivative(const double &input);


            /*!
             * Indicates that this function has a derivative.
             *
             * \return <code>true</code>, always.
             */
            virtual bool hasDerivative() {
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
