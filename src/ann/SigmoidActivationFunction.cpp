/*!
 * \file	SigmoidActivationFunction.cpp
 * \brief
 * \date	04.01.2013
 * \author	eveith
 */


#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


namespace Winzent
{
    namespace ANN
    {
        SigmoidActivationFunction::SigmoidActivationFunction(
                double scale,
                double transposition,
                QObject *parent):
                    ActivationFunction(scale, transposition, parent)
        {
        }


        ActivationFunction* SigmoidActivationFunction::clone() const
        {
            return new SigmoidActivationFunction(
                m_scalingFactor,
                m_transposition,
                parent());
        }


        double SigmoidActivationFunction::calculate(const double& input)
        {
            return (m_scalingFactor * (1.0 / (1.0 + std::exp(-input)))
                    + m_transposition);
        }


        double SigmoidActivationFunction::calculateDerivative(
                const double &input)
        {
            return (m_scalingFactor * input * (1.0 - input) + m_transposition);
        }
    }
}
