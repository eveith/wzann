/*
 * ActivationFunction.cpp
 *
 *  Created on: 16.10.2012
 *      Author: eveith
 */

#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {
        ActivationFunction::ActivationFunction(double steepness):
                m_steepness(steepness)
        {
        }


        ActivationFunction::~ActivationFunction()
        {
        }


        double ActivationFunction::steepness() const
        {
            return m_steepness;
        }


        double ActivationFunction::clip(
                double value,
                const double &min,
                const double &max)
                const
        {
            return (value < min) ? min : ((value > max) ? max : value);
        }
    }
}
