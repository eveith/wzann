/*!
 * \file	SigmoidActivationFunction.cpp
 * \brief
 * \date	04.01.2013
 * \author	eveith
 */


#include <QtDebug>

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
            qDebug()
                    << this
                    << "input" << input << "in" << in
                    << "="
                    << 1.0 / (1.0 + std::exp(-1.0 * steepness() * in));
            return 1.0 / (1.0 + std::exp(-1.0 * steepness() * in));
        }


        double SigmoidActivationFunction::calculateDerivative(
                const double &input)
        {
            double in = clip(input, 0.01, 0.99);
            qDebug() << this << "derivative" << input << "in" << in << "="
                << steepness() * in * (1.0 - in);
            return steepness() * in * (1.0 - in);
        }
    }
}
