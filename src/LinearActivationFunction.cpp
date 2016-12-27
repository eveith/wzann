#include <ClassRegistry.h>

#include "ActivationFunction.h"
#include "LinearActivationFunction.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {
        LinearActivationFunction::LinearActivationFunction(
                const qreal &steepness):
                    ActivationFunction(steepness)
        {
        }


        qreal LinearActivationFunction::calculate(const qreal &input)
        {
            return steepness() * input;
        }


        qreal LinearActivationFunction::calculateDerivative(
                const qreal &,
                const qreal &)
        {
            return steepness();
        }


        bool LinearActivationFunction::hasDerivative() const
        {
            return true;
        }


        ActivationFunction *LinearActivationFunction::clone() const
        {
            return new LinearActivationFunction(steepness());
        }


        bool LinearActivationFunction::equals(
                const ActivationFunction* const& other)
                const
        {
            return
                    nullptr != reinterpret_cast<
                            const LinearActivationFunction* const&>(other)
                        && ActivationFunction::equals(other);
        }
    } // namespace ANN
} // namespace Winzent


WINZENT_REGISTER_CLASS(
        Winzent::ANN::LinearActivationFunction,
        Winzent::ANN::ActivationFunction)
