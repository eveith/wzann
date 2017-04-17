#include "Neuron.h"
#include "WeightFixedException.h"

#include "Connection.h"


namespace Winzent {
    namespace ANN {
        Connection::Connection(
                Neuron& source,
                Neuron& destination,
                double weight):
                    m_weight(weight),
                    m_fixed(false),
                    m_sourceNeuron(&source),
                    m_destinationNeuron(&destination)
        {
        }


        double Connection::weight() const
        {
            return m_weight;
        }


        Connection& Connection::weight(double weight)
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


        Connection& Connection::fixedWeight(bool fixed)
        {
            m_fixed = fixed;
            return *this;
        }


        Neuron& Connection::source()
        {
            return *m_sourceNeuron;
        }


        Neuron const& Connection::source() const
        {
            return *m_sourceNeuron;
        }


        Connection& Connection::source(Neuron& source)
        {
            m_sourceNeuron = &source;
            return *this;
        }


        Neuron const& Connection::destination() const
        {
            return *m_destinationNeuron;
        }


        Neuron& Connection::destination()
        {
            return *m_destinationNeuron;
        }


        Connection& Connection::destination(Neuron& destination)
        {
            m_destinationNeuron = &destination;
            return *this;
        }


        double Connection::operator *(double rhs) const
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


        bool Connection::operator !=(const Connection& other) const
        {
            return !(*this == other);
        }
    } // namespace ANN
} // namespace Winzent
