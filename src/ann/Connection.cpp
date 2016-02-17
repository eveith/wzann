#include "Neuron.h"
#include "Exception.h"
#include "Connection.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {
        Connection::Connection(
                Neuron &source,
                Neuron &destination,
                const qreal &weight):
                    m_weight(weight),
                    m_fixed(false),
                    m_sourceNeuron(&source),
                    m_destinationNeuron(&destination)
        {
        }


        Connection *Connection::clone() const
        {
            Connection *clone = new Connection(
                    *m_sourceNeuron,
                    *m_destinationNeuron,
                    m_weight);
            clone->fixedWeight(fixedWeight());
            return clone;
        }


        qreal Connection::weight() const
        {
            return m_weight;
        }


        Connection &Connection::weight(const qreal &weight)
        {
            if (m_fixed) {
                throw WeightFixedException();
            } else {
                m_weight = weight;
            }

            return *this;
        }


        bool Connection::fixedWeight() const
        {
            return m_fixed;
        }


        Connection &Connection::fixedWeight(const bool &fixed)
        {
            m_fixed = fixed;
            return *this;
        }


        Neuron &Connection::source()
        {
            return *m_sourceNeuron;
        }


        const Neuron &Connection::source() const
        {
            return *m_sourceNeuron;
        }


        Connection &Connection::source(Neuron &source)
        {
            m_sourceNeuron = &source;
            return *this;
        }


        const Neuron &Connection::destination() const
        {
            return *m_destinationNeuron;
        }


        Neuron &Connection::destination()
        {
            return *m_destinationNeuron;
        }


        Connection &Connection::destination(Neuron &destination)
        {
            m_destinationNeuron = &destination;
            return *this;
        }


        qreal Connection::operator *(const qreal &rhs) const
        {
            return m_weight * rhs;
        }


        bool Connection::operator ==(const Connection& other) const
        {
            return (fixedWeight() == other.fixedWeight()
                    && 1.0 + weight() == 1.0 + other.weight()
                    && source() == other.source()
                    && destination() == other.destination());
        }


        bool Connection::equals(const Connection &other) const
        {
            return (fixedWeight() == other.fixedWeight()
                    && 1.0 + weight() == 1.0 + other.weight()
                    && source().equals(other.source())
                    && destination().equals(other.destination()));
        }


        bool Connection::operator !=(const Connection& other) const
        {
            return !(*this == other);
        }
    } // namespace ANN
} // namespace Winzent
