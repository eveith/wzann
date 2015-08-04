#include <QJsonObject>
#include <QJsonDocument>

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
            qreal ret = m_rememberedValue;
            m_rememberedValue = input * steepness();

            return ret;
        }


        ActivationFunction *RememberingActivationFunction::clone() const
        {
            RememberingActivationFunction *clone =
                    new RememberingActivationFunction(steepness());
            clone->m_rememberedValue = m_rememberedValue;

            return clone;
        }


        void RememberingActivationFunction::clear()
        {
            m_rememberedValue = 0.0;
        }


        QJsonDocument RememberingActivationFunction::toJSON() const
        {
            QJsonObject o = ActivationFunction::toJSON().object();

            o["rememberedValue"] = m_rememberedValue;

            return QJsonDocument(o);
        }


        void RememberingActivationFunction::fromJSON(
                const QJsonDocument &json)
        {
            ActivationFunction::fromJSON(json);
            m_rememberedValue = json.object()["rememberedValue"].toDouble();
        }


        bool RememberingActivationFunction::equals(
                const ActivationFunction* const &other)
                const
        {
            auto otherAF = reinterpret_cast<
                    const RememberingActivationFunction* const&>(other);
            return nullptr != otherAF
                    && m_rememberedValue == otherAF->m_rememberedValue
                    && ActivationFunction::equals(other);
        }
    }
}


WINZENT_REGISTER_CLASS(
        Winzent::ANN::RememberingActivationFunction,
        Winzent::ANN::ActivationFunction)
