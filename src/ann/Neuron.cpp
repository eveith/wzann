/*!
 * \file	Neuron.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include <QtDebug>

#include "ActivationFunction.h"
#include "Neuron.h"


namespace Winzent
{
    namespace ANN
    {
        Neuron::Neuron(ActivationFunction *activationFunction, QObject *parent):
            QObject(parent),
            m_activationFunction(activationFunction),
            m_lastInput(0.0),
            m_lastResult(0.0)
        {
        }


        Neuron::Neuron(const Neuron &rhs):
                QObject(rhs.parent()),
                m_activationFunction(rhs.m_activationFunction->clone()),
                m_lastInput(rhs.m_lastInput),
                m_lastResult(rhs.m_lastResult)
        {
        }


        Neuron::~Neuron()
        {
        }


        Neuron* Neuron::clone() const
        {
            Neuron *n = new Neuron(m_activationFunction->clone());

            n->m_lastResult = m_lastResult;
            n->m_lastInput = m_lastInput;
            n->setParent(parent());

            return n;
        }


        double Neuron::lastResult() const
        {
            return m_lastResult;
        }


        double Neuron::lastInput() const
        {
            return m_lastInput;
        }


        ActivationFunction* Neuron::activationFunction() const
        {
            return m_activationFunction;
        }


        double Neuron::activate(const double &sum)
        {
            m_lastInput = sum;
            m_lastResult = m_activationFunction->calculate(m_lastInput);

            qDebug() << this
                << "lastInput" << m_lastInput
                << "lastResult" << m_lastResult;

            return m_lastResult;
        }
    }
}
