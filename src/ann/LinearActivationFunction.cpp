#include <QObject>

#include "ActivationFunction.h"
#include "LinearActivationFunction.h"

namespace Winzent {
    namespace ANN {
        
        LinearActivationFunction::LinearActivationFunction(
                double scale,
                double transpose,
                QObject *parent):
                    ActivationFunction(scale, transpose, parent)
        {
        }


        double LinearActivationFunction::calculate(const double &input)
        {
            return m_scalingFactor * input + m_transposition;
        }


        double LinearActivationFunction::calculateDerivative(
                const double&)
        {
            return m_scalingFactor;
        }


        bool LinearActivationFunction::hasDerivative()
        {
            return true;
        }


        ActivationFunction *LinearActivationFunction::clone() const
        {
            return new LinearActivationFunction(
                        m_scalingFactor,
                        m_transposition,
                        parent());
        }
        
    } // namespace ANN
} // namespace Winzent
