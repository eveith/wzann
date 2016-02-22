#include <memory>

#include <QJsonObject>
#include <QJsonDocument>

#include <ClassRegistry.h>

#include "Layer.h"
#include "ActivationFunction.h"

#include "Neuron.h"
#include "Winzent-ANN_global.h"


using std::shared_ptr;


namespace Winzent {
    namespace ANN {
        Neuron::Neuron(ActivationFunction* const& activationFunction):
                m_parent(nullptr),
                m_activationFunction(activationFunction),
                m_lastInput(0.0),
                m_lastResult(0.0)
        {
        }


        Neuron::Neuron(shared_ptr<ActivationFunction> &activationFunction):
                m_parent(nullptr),
                m_activationFunction(activationFunction)
        {
        }


        Neuron::~Neuron()
        {
        }


        Neuron *Neuron::clone() const
        {
            Neuron *n = new Neuron(m_activationFunction->clone());

            n->m_lastInput = m_lastInput;
            n->m_lastResult = m_lastResult;
            n->m_parent = m_parent;

            return n;
        }


        Layer *Neuron::parent() const
        {
            return m_parent;
        }


        qreal Neuron::lastResult() const
        {
            return m_lastResult;
        }


        qreal Neuron::lastInput() const
        {
            return m_lastInput;
        }


        ActivationFunction *Neuron::activationFunction() const
        {
            return m_activationFunction.get();
        }


        Neuron &Neuron::activationFunction(
                ActivationFunction *const &activationFunction)
        {
            m_activationFunction.reset(activationFunction);
            return *this;
        }


        Neuron &Neuron::activationFunction(
                shared_ptr<ActivationFunction> &activationFunction)
        {
            m_activationFunction = activationFunction;
            return *this;
        }


        qreal Neuron::activate(const qreal &sum)
        {
            m_lastInput = sum;
            m_lastResult = m_activationFunction->calculate(sum);
            return m_lastResult;
        }


        void Neuron::clear()
        {
            m_lastInput = m_lastResult = 0.0;
            m_activationFunction.reset();
        }


        QJsonDocument Neuron::toJSON() const
        {
            QJsonObject o;

            o["lastInput"] = lastInput();
            o["lastResult"] = lastResult();
            o["activationFunction"] = activationFunction()->toJSON().object();

            return QJsonDocument(o);
        }


        void Neuron::fromJSON(const QJsonDocument& json)
        {
            clear();
            QJsonObject o = json.object();

            m_lastInput = o["lastInput"].toDouble();
            m_lastResult = o["lastResult"].toDouble();
            m_activationFunction.reset(
                    ClassRegistry<ActivationFunction>::instance()->create(
                        o["activationFunction"]
                            .toObject()["type"].toString()));
            activationFunction()->fromJSON(QJsonDocument(
                    o["activationFunction"].toObject()));
        }


        bool Neuron::equals(const Neuron &other) const
        {
            return (this->lastInput() == other.lastInput()
                    && this->lastResult() == other.lastResult()
                    && this->activationFunction()->equals(
                        other.activationFunction()));
        }


        bool Neuron::operator ==(const Neuron& other) const
        {
            return this == &other;
        }


        bool Neuron::operator !=(const Neuron& other) const
        {
            return !(*this == other);
        }
    }
}
