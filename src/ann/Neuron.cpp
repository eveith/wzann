/*!
 * \file	Neuron.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include <memory>

#include "Layer.h"
#include "ActivationFunction.h"

#include "Neuron.h"


using std::shared_ptr;


namespace Winzent {
    namespace ANN {


        Neuron::Neuron(ActivationFunction *activationFunction):
                m_parent(nullptr),
                m_activationFunction(activationFunction)
        {
        }


        Neuron::Neuron(shared_ptr<ActivationFunction> &activationFunction):
                m_parent(nullptr),
                m_activationFunction(activationFunction)
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


        const QVector<qreal> Neuron::lastInputs() const
        {
            return QVector<qreal>({ m_lastInput });
        }


        qreal Neuron::lastInput() const
        {
            return m_lastInput;
        }


        const QVector<qreal> Neuron::lastResults() const
        {
            return QVector<qreal>({ m_lastResult });
        }


        int Neuron::cacheSize() const
        {
            return 1;
        }


        Neuron &Neuron::cacheSize(const int &)
        {
            return *this;
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
    }
}
