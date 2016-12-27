#ifndef WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
#define WINZENT_ANN_LINEARACTIVATIONFUNCTION_H


#include <QObject>

#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        /*!
         * \brief The linear activation function resembles
         *  a simple linear function.
         *
         * Basically, this function is the equivalent of f(x) = x, with an
         * optional scaling and transposition applied, so that is can become
         * f(x) = ax + c.
         *
         * The typical use case for this activation function is the input layer.
         */
        class WINZENTANNSHARED_EXPORT LinearActivationFunction:
                public ActivationFunction
        {
            Q_OBJECT
        public:


            /*!
             * \brief Creates a new linear activation function with
             *  an optional scaling and transposition.
             *
             * \param steepness Steepness the function: f(x) = steepness * x
             */
            LinearActivationFunction(const qreal &steepness = 1.0);


            /*!
             * \return `input * steepness()`
             */
            virtual qreal calculate(const qreal &input) override;


            /*!
             * \return ActivationFunction#steepness()
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
             * \brief Checks for equality
             *
             * \param[in] other Another ActivationFunction object
             *
             * \return True iff the other object is of the same class and
             *  has the same parameters.
             */
            virtual bool equals(const ActivationFunction* const& other) const
                override;
        };
    } // namespace ANN
} // namespace Winzent


#endif // WINZENT_ANN_LINEARACTIVATIONFUNCTION_H
