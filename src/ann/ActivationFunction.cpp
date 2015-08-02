#include <QObject>

#include "ActivationFunction.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {
        ActivationFunction::ActivationFunction(const qreal &steepness):
                QObject(),
                m_steepness(steepness)
        {
        }


        ActivationFunction::~ActivationFunction()
        {
        }


        qreal ActivationFunction::steepness() const
        {
            return m_steepness;
        }


        qreal ActivationFunction::clip(
                const qreal &value,
                const qreal &min,
                const qreal &max)
                const
        {
            return (value < min) ? min : ((value > max) ? max : value);
        }
    }
}
