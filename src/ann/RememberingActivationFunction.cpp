/*!
 * \file	RememberingActivationFunction.cpp
 * \brief
 * \date	04.01.2013
 * \author	eveith
 */


#include "RememberingActivationFunction.h"


namespace Winzent
{
    namespace ANN
    {
        RememberingActivationFunction::~RememberingActivationFunction()
        {
        }


        double RememberingActivationFunction::calculate(const double &input)
        {
            double ret = m_remeberedValue;
            m_remeberedValue = input;

            return ret;
        }


        ActivationFunction* RememberingActivationFunction::clone() const
        {
            return new RememberingActivationFunction(m_remeberedValue);
        }
    }
}
