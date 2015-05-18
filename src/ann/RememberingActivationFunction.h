/*
 * RememberingActivationFunction.h
 *
 *  Created on: 05.11.2012
 *      Author: eveith
 */

#ifndef REMEMBERINGACTIVATIONFUNCTION_H_
#define REMEMBERINGACTIVATIONFUNCTION_H_

#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {

        /*!
         * \brief An activation function that remembers its last value
         *
         * Represents an activation function which always remembers
         * the last value. This functionality is used, e.g., in
         * Elman Networks, where the context layer always remembers
         * the last output of the hidden layer.
         *
         * \sa ElmanNetwork
         */
        class RememberingActivationFunction: public ActivationFunction
        {

        private:


            /*!
             * The saved/remembered value
             */
            qreal m_remeberedValue;


        public:


            /*!
             * Constructs a new instance of this activation function
             * and initializes the remembered value with 0.0.
             */
            RememberingActivationFunction(qreal steepness = 1.0);


            /*!
             * \brief Returns the lastly remembered value and replaces it with
             *  the newly supplied one.
             *
             * \param[in] input The new value to remember
             *
             * \return The last value that had been remembered
             */
            virtual qreal calculate(const qreal &input) override;


            /*!
             * Returns the derivative. A remembering activation function does
             * not really have a derivative, so it is treated as if a simple
             * \f$f'(x) = \frac{d}{dx}ax = a\f$, with _a_ being the
             * ActivationFunction#steepness.
             *
             * \return ActivationFunction#steepness()
             *
             * \sa ActivationFunction#steepness
             */
            virtual qreal calculateDerivative(const qreal &, const qreal &)
                    override {
                return steepness();
            }


            /*!
             * Indicates that this activation function has a
             * derivative, albeit not a very useful one.
             *
             * \return `true`
             */
            virtual bool hasDerivative() const override {
                return true;
            }


            /*!
             * \brief Clones the activation function
             *
             * Creates a new instance of this activation function and
             * initializes it with the currently remembered value.
             *
             * \return A new
             *  <code>RememberingActivationFunction</code> instance
             *  with the same remembered value as this one.
             */
            virtual ActivationFunction *clone() const override;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* REMEMBERINGACTIVATIONFUNCTION_H_ */
