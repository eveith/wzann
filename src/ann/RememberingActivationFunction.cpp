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


        void RememberingActivationFunction::clear()
        {
            m_remeberedValue = 0.0;
        }


        QJsonDocument RememberingActivationFunction::toJSON() const
        {
            QJsonObject o = ActivationFunction::toJSON().object();

            o["rememberedValue"] = m_remeberedValue;

            return QJsonDocument(o);
        }


        void RememberingActivationFunction::fromJSON(
                const QJsonDocument &json)
        {
            ActivationFunction::fromJSON(json);
            m_remeberedValue = json.object()["rememberedValue"].toDouble();
        }


        bool RememberingActivationFunction::operator ==(
                const RememberingActivationFunction &other)
                const
        {
            return steepness() == other.steepness()
                    && m_remeberedValue == other.m_remeberedValue;
        }
    }
}


WINZENT_REGISTER_CLASS(
        Winzent::ANN::RememberingActivationFunction,
        Winzent::ANN::ActivationFunction)
