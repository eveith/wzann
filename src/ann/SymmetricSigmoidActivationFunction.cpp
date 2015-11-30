#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"
#include "SymmetricSigmoidActivationFunction.h"

namespace Winzent {
    namespace ANN {
        SymmetricSigmoidActivationFunction::
        SymmetricSigmoidActivationFunction(const qreal &steepness):
                ActivationFunction(steepness)
        {
        }


        SymmetricSigmoidActivationFunction::
        ~SymmetricSigmoidActivationFunction()
        {
        }


        ActivationFunction *SymmetricSigmoidActivationFunction::clone() const
        {
            return new SymmetricSigmoidActivationFunction(steepness());
        }


        qreal SymmetricSigmoidActivationFunction::calculate(
                const qreal &input)
        {
            auto in = clip(input, -45.0f / steepness(), 45.0f / steepness());
            return 2.0 * (-0.5+1.0/(1.0 + std::exp(-1.0 * steepness() * in)));
        }


        bool SymmetricSigmoidActivationFunction::hasDerivative() const
        {
            return true;
        }


        qreal SymmetricSigmoidActivationFunction::calculateDerivative(
                const qreal &,
                const qreal &result)
        {
            auto in = clip(result, 0.01f, 0.99f);
            return (2.0 * steepness() * std::exp(-steepness() * in))
                    / std::pow(1.0+std::exp(-steepness() * in), 2);
        }


        bool SymmetricSigmoidActivationFunction::equals(
                const ActivationFunction* const& other)
                const
        {
            return nullptr != other
                    && nullptr != reinterpret_cast<
                        const SymmetricSigmoidActivationFunction* const&>(
                            other)
                    && ActivationFunction::equals(other);
        }

    } // namespace ANN
} // namespace Winzent

