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
            RememberingActivationFunction(): m_remeberedValue(0.0) {}


            /*!
             * Constructs a new instance of this activation function
             * and initializes the remembered value with a supplied
             * one.
             */
            RememberingActivationFunction(double initialValue):
                    m_remeberedValue(initialValue)
            {}


            virtual ~RememberingActivationFunction();


            virtual double calculate(const double &input);


            /*!
             * The remembering activation function has no derivative.
             * As a consequence, calling this function will throw an
             * exception.
             */
            virtual double calculateDerivative(const double&) {
                throw "No derivative available";
                return 0.0;
            }


            /*!
             * Indicates that this activation function has no
             * derivative.
             *
             * \return <code>false</code>, always.
             */
            virtual bool hasDerivative() {
                return false;
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
