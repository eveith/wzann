#include "Neuron.h"
#include "Exception.h"

#include "Connection.h"


namespace Winzent {
    namespace ANN {
        Connection::Connection(
                Neuron *const &source,
                Neuron *const &destination,
                const double &weight):
                    m_weight(weight),
                    m_fixed(false),
                    m_sourceNeuron(source),
                    m_destinationNeuron(destination)
        {
        }


        Connection *Connection::clone() const
        {
            Connection *clone = new Connection(
                    m_sourceNeuron,
                    m_destinationNeuron,
                    m_weight);
            clone->fixedWeight(fixedWeight());
            return clone;
        }


        double Connection::weight() const
        {
            return m_weight;
        }


        Connection &Connection::weight(const double &weight)
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


        void Connection::fixedWeight(const bool &fixed)
        {
            m_fixed = fixed;
        }


        Neuron *Connection::source() const
        {
            return m_sourceNeuron;
        }


        Connection &Connection::source(Neuron *const &source)
        {
            m_sourceNeuron = source;
            return *this;
        }


        Neuron *Connection::destination() const
        {
            return m_destinationNeuron;
        }


        Connection &Connection::destination(Neuron *const &destination)
        {
            m_destinationNeuron = destination;
            return *this;
        }


        double Connection::operator *(const double &rhs) const
        {
            return m_weight * rhs;
        }
    } // namespace ANN
} // namespace Winzent
