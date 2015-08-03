#ifndef SIGMOIDACTIVATIONFUNCTION_H_
#define SIGMOIDACTIVATIONFUNCTION_H_


#include <QObject>

#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        /*!
         * \brief The SigmoidActivationFunction class represents an activation
         *  function that follows a sigmoid pattern
         */
        class WINZENTANNSHARED_EXPORT SigmoidActivationFunction:
                public ActivationFunction
        {
            Q_OBJECT

        public:


            /*!
             * \brief Constructs a new SigmoidActivationFunction
             *
             * \param[in] steepness The new function's steepness
             */
            SigmoidActivationFunction(const qreal &steepness = 1.0);


            /*!
             * \brief Calculates output using the sigmoid function.
             */
            virtual qreal calculate(const qreal &input) override;


            /*!
             * Calculates the output of the derivative of the sigmoid
             * function.
             *
             * \sa ActivationFunction#calculateDerivative
             */
            virtual qreal calculateDerivative(
                    const qreal &,
                    const qreal &result)
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


            //! Checks for equality
            bool operator ==(const SigmoidActivationFunction &other) const;
        };
    } /* namespace ANN */
} /* namespace Winzent */


#endif /* SIGMOIDACTIVATIONFUNCTION_H_ */
