/*!
 * \file	Neuron.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include "Layer.h"
#include "ActivationFunction.h"

#include "Neuron.h"


namespace Winzent {
    namespace ANN {


        Neuron::Neuron(
                ActivationFunction *activationFunction,
                QObject *parent):
                    QObject(parent),
                    m_activationFunction(activationFunction)
        {
            m_activationFunction->setParent(this);
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
            return m_activationFunction;
        }


        qreal Neuron::activate(const qreal &sum)
        {
            m_lastInput = sum;
            m_lastResult = m_activationFunction->calculate(sum);
            return m_lastResult;
        }
    }
}
