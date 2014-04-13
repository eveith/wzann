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
                double steepness,
                QObject *parent):
                    ActivationFunction(steepness, parent)
        {
        }


        ActivationFunction* SigmoidActivationFunction::clone() const
        {
            return new SigmoidActivationFunction(steepness(), parent());
        }


        double SigmoidActivationFunction::calculate(const double& input)
        {
            double in = clip(input, -45/steepness(), 45/steepness());
            return 1.0 / (1.0 + std::exp(-1.0 * steepness() * in));
        }


        double SigmoidActivationFunction::calculateDerivative(
                const double &,
                const double &result)
        {
            double in = clip(result, 0.01, 0.99);
            return steepness() * in * (1.0 - in);
        }
    }
}
