/*!
 * \file	SigmoidActivationFunction.cpp
 * \brief
 * \date	04.01.2013
 * \author	eveith
 */


#include <cmath>

#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


namespace Winzent {
    namespace ANN {
        SigmoidActivationFunction::SigmoidActivationFunction(double steepness):
                ActivationFunction(steepness)
        {
        }


        ActivationFunction *SigmoidActivationFunction::clone() const
        {
            return new SigmoidActivationFunction(steepness());
        }


        double SigmoidActivationFunction::calculate(const double &input)
        {
            auto in = clip(input, -45.0f / steepness(), 45.0f / steepness());
            return 1.0f / (1.0f + std::exp(-1.0f * steepness() * in));
        }


        double SigmoidActivationFunction::calculateDerivative(
                const double &,
                const double &result)
        {
            auto in = clip(result, 0.01f, 0.99f);
            return steepness() * in * (1.0f - in);
        }
    }
}
