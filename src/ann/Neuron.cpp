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
            m_lastInputs(QVector<qreal>()),
            m_lastResults(QVector<qreal>()),
            m_cacheSize(0)
        {
        }


        Neuron::Neuron(const Neuron &rhs):
                QObject(rhs.parent()),
                m_activationFunction(rhs.m_activationFunction->clone()),
                m_lastInputs(QVector<qreal>(rhs.m_lastInputs)),
                m_lastResults(QVector<qreal>(rhs.m_lastResults)),
                m_cacheSize(rhs.m_cacheSize)
        {
        }


        Neuron *Neuron::clone() const
        {
            Neuron *n = new Neuron(m_activationFunction->clone());

            n->m_lastInputs = QVector<qreal>(m_lastInputs);
            n->m_lastResults = QVector<qreal>(m_lastResults);
            n->m_cacheSize = m_cacheSize;
            n->setParent(parent());

            return n;
        }


        qreal Neuron::lastResult() const
        {
            return m_lastResults.first();
        }


        const QVector<qreal> Neuron::lastInputs() const
        {
            return m_lastInputs;
        }


        qreal Neuron::lastInput() const
        {
            return m_lastInputs.first();
        }


        const QVector<qreal> Neuron::lastResults() const
        {
            return m_lastResults;
        }


        int Neuron::cacheSize() const
        {
            return m_cacheSize;
        }


        Neuron *Neuron::cacheSize(int cacheSize)
        {
            int oldCacheSize = m_cacheSize;
            m_cacheSize = cacheSize;
            if (oldCacheSize > cacheSize) {
                trimCache();
            }

            return this;
        }


        void Neuron::trimCache()
        {
            m_lastInputs.resize(cacheSize());
            m_lastResults.resize(cacheSize());
        }


        ActivationFunction* Neuron::activationFunction() const
        {
            return m_activationFunction;
        }


        qreal Neuron::activate(const qreal &sum)
        {
            qreal result = m_activationFunction->calculate(sum);

            if (cacheSize() > 0) {
                m_lastInputs.push_front(qreal(sum));
                m_lastResults.push_front(qreal(result));

                if (m_lastInputs.size() > cacheSize()) {
                    trimCache();
                }

                qDebug() << this
                    << "lastInput" << m_lastInputs.first()
                    << "lastResult" << m_lastResults.first();
            }

            return result;
        }
    }
}
