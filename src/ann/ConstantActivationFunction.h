#ifndef WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
#define WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H


#include <QObject>
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
            Q_OBJECT

        private:


            /*!
             * The constant value we emit
             */
            double m_value;


        public:


            /*!
             * Creates a new instance of this class with a constant value of
             * <code>1.0</code>, which is optionally configurable.
             *
             * \param value The constant activation value, default to 1.0.
             */
            ConstantActivationFunction(double value = 1.0, QObject *parent = 0);


            virtual double calculate(const double &);


            virtual double calculateDerivative(const double &);


            virtual bool hasDerivative();


            virtual ActivationFunction* clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONSTANTACTIVATIONFUNCTION_H
