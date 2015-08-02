#include <QObject>

#include "ActivationFunction.h"
#include "ConstantActivationFunction.h"


namespace Winzent {
    namespace ANN {
        ConstantActivationFunction::ConstantActivationFunction(qreal value):
                ActivationFunction(1.0),
                m_value(value)
        {
        }


        qreal ConstantActivationFunction::calculate(const qreal &)
        {
            return m_value;
        }


        qreal ConstantActivationFunction::calculateDerivative(
                const qreal &,
                const qreal &)
        {
            return 0.0;
        }


        bool ConstantActivationFunction::hasDerivative() const
        {
            return true;
        }


        ActivationFunction *ConstantActivationFunction::clone() const
        {
            return new ConstantActivationFunction(m_value);
        }
    } // namespace ANN
} // namespace Winzent
