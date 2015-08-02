#include <ClassRegistry.h>

#include "ActivationFunction.h"
#include "RememberingActivationFunction.h"


namespace Winzent {
    namespace ANN {
        RememberingActivationFunction::RememberingActivationFunction(
                const qreal &steepness):
                    ActivationFunction(steepness)
        {
        }


        qreal RememberingActivationFunction::calculate(const qreal &input)
        {
            qreal ret = m_remeberedValue;
            m_remeberedValue = input * steepness();

            return ret;
        }


        ActivationFunction *RememberingActivationFunction::clone() const
        {
            RememberingActivationFunction *clone =
                    new RememberingActivationFunction(steepness());
            clone->m_remeberedValue = m_remeberedValue;

            return clone;
        }
    }
}


WINZENT_REGISTER_CLASS(
        Winzent::ANN::RememberingActivationFunction,
        Winzent::ANN::ActivationFunction)
