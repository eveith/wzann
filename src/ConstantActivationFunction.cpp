#include <ClassRegistry.h>

#include "ActivationFunction.h"
#include "ConstantActivationFunction.h"


namespace Winzent {
    namespace ANN {
        ConstantActivationFunction::ConstantActivationFunction(
                const qreal& value):
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


        bool ConstantActivationFunction::equals(
                const ActivationFunction* const& other)
                const
        {
            auto otherAF = reinterpret_cast<
                    const ConstantActivationFunction* const&>(other);
            return nullptr != otherAF
                    && m_value == otherAF->m_value
                    && ActivationFunction::equals(other);
        }
    } // namespace ANN
} // namespace Winzent


WINZENT_REGISTER_CLASS(
        Winzent::ANN::ConstantActivationFunction,
        Winzent::ANN::ActivationFunction)
