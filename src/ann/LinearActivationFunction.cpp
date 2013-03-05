#include <QObject>

#include "ActivationFunction.h"
#include "LinearActivationFunction.h"

namespace Winzent {
    namespace ANN {
        
        LinearActivationFunction::LinearActivationFunction(
                double steepness,
                QObject *parent):
                    ActivationFunction(steepness, parent)
        {
        }


        double LinearActivationFunction::calculate(const double &input)
        {
            return steepness() * input;
        }


        double LinearActivationFunction::calculateDerivative(
                const double&,
                const double &)
        {
            return steepness();
        }


        bool LinearActivationFunction::hasDerivative() const
        {
            return true;
        }


        ActivationFunction *LinearActivationFunction::clone() const
        {
            return new LinearActivationFunction(steepness(), parent());
        }
        
    } // namespace ANN
} // namespace Winzent
