#ifndef WINZENT_ANN_TANHACTIVATIONFUNCTION_H
#define WINZENT_ANN_TANHACTIVATIONFUNCTION_H


#include <ClassRegistry.h>

#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        /*!
         * \brief The TanhActivationFunction class implements a
         *  tanh(x) function with result interval (-1.0; 1.0).
         */
        class WINZENTANNSHARED_EXPORT TanhActivationFunction:
                public ActivationFunction
        {
            Q_OBJECT

        public:


            /*!
             * \brief Constructs a new sigsym object
             *
             * \param[in] steepness The function's steepness
             */
            TanhActivationFunction(const qreal &steepness = 1.0);


            virtual ~TanhActivationFunction();


            /*!
             * \brief Clones this object
             *
             * \return A clone of `this`
             */
            virtual ActivationFunction *clone() const override;


            /*!
             * \brief Calculated sigsym(input)
             *
             * \param[in] input The input X value
             */
            virtual qreal calculate(const qreal &input) override;


            /*!
             * \brief Indiciates that this activation function has a
             *  derivative
             *
             * \return  `true`
             */
            virtual bool hasDerivative() const override;

            /*!
             * \brief Calculates sigsym(x) dx
             *
             * \param[in] sum The neuron's last input from the net
             *  (\f$net_j\f$)
             *
             * \param[in] result The neuron's output (\f$\varphi{}(net_j)\f$)
             */
            virtual qreal calculateDerivative(
                    const qreal &sum,
                    const qreal &result)
                    override;


            /*!
             * \brief Checks for equality of two Activation Function objects
             *
             * \param[in] other Another ActivationFunction object
             *
             * \return `true` if the other object is also a
             *  TanhActivationFunction, and has the same
             *  steepness.
             */
            virtual bool equals(const ActivationFunction *const &other)
                    const override;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_TANHACTIVATIONFUNCTION_H
