/*
 * ActivationFunction.cpp
 *
 *  Created on: 16.10.2012
 *      Author: eveith
 */

#include "ActivationFunction.h"


namespace Winzent
{
    namespace ANN
    {
        ActivationFunction::ActivationFunction(
                double scale,
                double transpose,
                QObject *parent):
                    QObject(parent),
                    m_scalingFactor(scale),
                    m_transposition(transpose)
        {
        }
    }
}
