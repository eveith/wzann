#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"
#include "TanhActivationFunction.h"


namespace Winzent {
    namespace ANN {
        TanhActivationFunction::
        TanhActivationFunction(const qreal &steepness):
                ActivationFunction(steepness)
        {
        }


        TanhActivationFunction::~TanhActivationFunction()
        {
        }


        ActivationFunction *TanhActivationFunction::clone() const
        {
            return new TanhActivationFunction(steepness());
        }


        qreal TanhActivationFunction::calculate(const qreal &input)
        {
            return std::tanh(input) * steepness();
        }


        bool TanhActivationFunction::hasDerivative() const
        {
            return true;
        }


        qreal TanhActivationFunction::calculateDerivative(
                const qreal &,
                const qreal &result)
        {
            return (1 - std::pow(std::tanh(result), 2)) * steepness();
        }


        bool TanhActivationFunction::equals(
                const ActivationFunction *const &other)
                const
        {
            return nullptr != other
                    && nullptr != reinterpret_cast<
                        const TanhActivationFunction* const&>(
                            other)
                    && ActivationFunction::equals(other);
        }
    } // namespace ANN
} // namespace Winzent


WINZENT_REGISTER_CLASS(
        Winzent::ANN::TanhActivationFunction,
        Winzent::ANN::ActivationFunction)
