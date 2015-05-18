/*
 * ActivationFunction.cpp
 *
 *  Created on: 16.10.2012
 *      Author: eveith
 */

#include "ActivationFunction.h"


namespace Winzent {
    namespace ANN {
        ActivationFunction::ActivationFunction(qreal steepness):
                m_steepness(steepness)
        {
        }


        qreal ActivationFunction::steepness() const
        {
            return m_steepness;
        }


        qreal ActivationFunction::clip(
                qreal value,
                const qreal &min,
                const qreal &max)
                const
        {
            return (value < min) ? min : ((value > max) ? max : value);
        }
    }
}
