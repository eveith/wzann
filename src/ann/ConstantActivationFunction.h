#ifndef WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
#define WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H


#include <QObject>

#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"


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
            Q_OBJECT

            friend bool ActivationFunction::equals(
                const ActivationFunction* const&)
                const;

        public:


            /*!
             * \brief Creates a new instance of this class.
             *
             * \param value The constant activation value; defaults to 1.0.
             */
            ConstantActivationFunction(const qreal& value = 1.0);


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


            /*!
             * \brief Checks for equality of two ActivationFunction objects
             *
             * \param[in] other The other ActivationFunction object
             *
             * \return True if `other` is also a ConstantActivationFunction
             *  and has the same parameters.
             */
            virtual bool equals(const ActivationFunction* const& other) const
                override;


        private:


            /*!
             * The constant value we emit
             */
            qreal m_value;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
