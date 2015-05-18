#ifndef WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
#define WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H


#include <QtGlobal>

#include "ActivationFunction.h"

namespace Winzent {
    namespace ANN {
        

        /*!
         * Represents an activation function which always emits a constant
         * value.
         *
         * The typical use case of this activation function would be a bias
         * neuron, which will emit 1.0 every time it is activated.
         */
        class ConstantActivationFunction: public ActivationFunction
        {

        private:


            /*!
             * The constant value we emit
             */
            qreal m_value;


        public:


            /*!
             * Creates a new instance of this class with a constant value of
             * <code>1.0</code>, which is optionally configurable.
             *
             * \param value The constant activation value; defaults to 1.0.
             */
            ConstantActivationFunction(qreal value = 1.0);


            /*!
             * \return The constant value the function was created with
             *
             * \sa #m_value
             */
            virtual qreal calculate(const qreal &) override;


            /*!
             * \return 0.0
             */
            virtual qreal calculateDerivative(const qreal &, const qreal &)
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

#endif // WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
