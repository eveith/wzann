/*
 * SigmoidActivationFunction.h
 *
 *  Created on: 04.11.2012
 *      Author: eveith
 */

#ifndef SIGMOIDACTIVATIONFUNCTION_H_
#define SIGMOIDACTIVATIONFUNCTION_H_


#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {


        /*!
         * \brief The SigmoidActivationFunction class represents an activation
         *  function that follows a sigmoid pattern
         */
        class SigmoidActivationFunction: public ActivationFunction
        {
        public:


            /*!
             * \brief Constructs a new SigmoidActivationFunction
             *
             * \param[in] steepness The new function's steepness
             */
            SigmoidActivationFunction(double steepness = 1.0);


            /*!
             * \brief Calculates output using the sigmoid function.
             */
            virtual double calculate(const double &input) override;


            /*!
             * Calculates the output of the derivative of the sigmoid
             * function.
             *
             * \sa ActivationFunction#calculateDerivative
             */
            virtual double calculateDerivative(
                    const double &,
                    const double &result)
                    override;


            /*!
             * \brief Indicates that this function has a derivative.
             *
             * \return `true`
             */
            virtual bool hasDerivative() const override {
                return true;
            }


            /*!
             * Clones this activation function and returns a new
             * instance.
             *
             * \return A new instance of this class
             */
            virtual ActivationFunction *clone() const override;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* SIGMOIDACTIVATIONFUNCTION_H_ */
