#include <QObject>

#include <QJsonObject>
#include <QJsonDocument>

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


        void ActivationFunction::clear()
        {
        }


        QJsonDocument ActivationFunction::toJSON() const
        {
            QJsonObject o;

            o["type"] = metaObject()->className();
            o["steepness"] = m_steepness;

            return QJsonDocument(o);
        }


        void ActivationFunction::fromJSON(const QJsonDocument &json)
        {
            QJsonObject o = json.object();
            m_steepness = o["steepness"].toDouble();
        }


        bool ActivationFunction::equals(
                const ActivationFunction* const& other)
                const
        {
            return steepness() == other->steepness();
        }
    }
}
