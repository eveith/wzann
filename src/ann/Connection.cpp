#include <QObject>

#include "Neuron.h"
#include "Exception.h"

#include "Connection.h"


namespace Winzent {
    namespace ANN {


        Connection::Connection(
                Neuron *source,
                Neuron *destination,
                double weight,
                QObject *parent):
                    QObject(parent),
                    m_weight(weight),
                    m_fixed(false),
                    m_sourceNeuron(source),
                    m_destinationNeuron(destination)
        {
        }


        double Connection::weight() const
        {
            return m_weight;
        }


        void Connection::weight(double weight) throw(WeightFixedException)
        {
            if (m_fixed) {
                throw WeightFixedException();
            } else {
                m_weight = weight;
            }
        }


        bool Connection::fixedWeight() const
        {
            return m_fixed;
        }


        void Connection::fixedWeight(bool fixed)
        {
            m_fixed = fixed;
        }


        Neuron *Connection::source() const
        {
            return m_sourceNeuron;
        }


        void Connection::source(Neuron *source)
        {
            m_sourceNeuron = source;
        }


        Neuron *Connection::destination() const
        {
            return m_destinationNeuron;
        }


        void Connection::destination(Neuron *destination)
        {
            m_destinationNeuron = destination;
        }


        double Connection::operator *(const double &rhs) const
        {
            return m_weight * rhs;
        }
    } // namespace ANN
} // namespace Winzent
