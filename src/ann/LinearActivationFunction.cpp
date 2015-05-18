#include <QObject>

#include "ActivationFunction.h"
#include "LinearActivationFunction.h"

namespace Winzent {
    namespace ANN {
        LinearActivationFunction::LinearActivationFunction(qreal steepness):
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
    } // namespace ANN
} // namespace Winzent
