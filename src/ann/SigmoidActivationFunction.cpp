/*!
 * \file	SigmoidActivationFunction.cpp
 * \brief
 * \date	04.01.2013
 * \author	eveith
 */


#include <QtGlobal>

#include <cmath>

#include "ActivationFunction.h"
#include "SigmoidActivationFunction.h"


namespace Winzent {
    namespace ANN {
        SigmoidActivationFunction::SigmoidActivationFunction(qreal steepness):
                ActivationFunction(steepness)
        {
        }


        ActivationFunction *SigmoidActivationFunction::clone() const
        {
            return new SigmoidActivationFunction(steepness());
        }


        qreal SigmoidActivationFunction::calculate(const qreal &input)
        {
            auto in = clip(input, -45.0f / steepness(), 45.0f / steepness());
            return 1.0f / (1.0f + std::exp(-1.0f * steepness() * in));
        }


        qreal SigmoidActivationFunction::calculateDerivative(
                const qreal &,
                const qreal &result)
        {
            auto in = clip(result, 0.01f, 0.99f);
            return steepness() * in * (1.0f - in);
        }
    }
}
