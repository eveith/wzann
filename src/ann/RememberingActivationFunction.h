/*
 * RememberingActivationFunction.h
 *
 *  Created on: 05.11.2012
 *      Author: eveith
 */

#ifndef REMEMBERINGACTIVATIONFUNCTION_H_
#define REMEMBERINGACTIVATIONFUNCTION_H_

#include "ActivationFunction.h"


namespace Winzent
{
    namespace ANN
    {

        /*!
         * Represents an activation function which always remembers
         * the last value. This functionality is used, e.g., in
         * Elman Networks, where the context layer always remembers
         * the last output of the hidden layer.
         *
         * \sa ElmanNetwork
         */
        class RememberingActivationFunction: public ActivationFunction
        {
            Q_OBJECT

        private:


            /*!
             * The saved/remembered value
             */
            double m_remeberedValue;


        public:


            /*!
             * Constructs a new instance of this activation function
             * and initializes the remembered value with 0.0.
             */
            RememberingActivationFunction(
                    double steepness = 1.0,
                    QObject *parent = 0);


            virtual ~RememberingActivationFunction();


            virtual double calculate(const double &input);


            /*!
             * Returns the derivative. A remembering activation function does
             * not really have a derivative, so it is treated as if a simple
             * <code>f'(x) = d/dx(ax)</code>.
             */
            virtual double calculateDerivative(const double&) {
                return steepness();
            }


            /*!
             * Indicates that this activation function has no
             * derivative.
             *
             * \return <code>true</code>, always.
             */
            virtual bool hasDerivative() const {
                return true;
            }


            /*!
             * Creates a new instance of this activation function and
             * initializes it with the currently remembered value.
             *
             * \return A new
             *  <code>RememberingActivationFunction</code> instance
             *  with the same remembered value as this one.
             */
            virtual ActivationFunction* clone() const;
        };

    } /* namespace ANN */
} /* namespace Winzent */

#endif /* REMEMBERINGACTIVATIONFUNCTION_H_ */
