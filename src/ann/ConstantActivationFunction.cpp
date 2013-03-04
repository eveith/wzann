#include <QObject>

#include "ActivationFunction.h"
#include "ConstantActivationFunction.h"


namespace Winzent {
    namespace ANN {
        
        ConstantActivationFunction::ConstantActivationFunction(
                double value,
                QObject *parent):
                    ActivationFunction(1.0, parent),
                    m_value(value)
        {
        }


        double ConstantActivationFunction::calculate(const double &)
        {
            return m_value;
        }


        double ConstantActivationFunction::calculateDerivative(const double &)
        {
            return 0.0;
        }


        bool ConstantActivationFunction::hasDerivative() const
        {
            return true;
        }


        ActivationFunction* ConstantActivationFunction::clone() const
        {
            return new ConstantActivationFunction(m_value, parent());
        }
    } // namespace ANN
} // namespace Winzent
