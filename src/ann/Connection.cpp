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


        Connection::Connection(const Connection &rhs):
                QObject(rhs.parent()),
                m_weight(rhs.m_weight),
                m_fixed(rhs.m_fixed),
                m_sourceNeuron(rhs.m_sourceNeuron),
                m_destinationNeuron(rhs.m_destinationNeuron)
        {
        }


        Connection *Connection::clone() const
        {
            Connection *clone = new Connection(
                    m_sourceNeuron,
                    m_destinationNeuron,
                    m_weight,
                    parent());
            clone->fixedWeight(fixedWeight());
            return clone;
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



        double Connection::setRandomWeight(const double &min, const double &max)
                throw(WeightFixedException)
        {
            weight(min + (qrand() * abs(max-min)
                    / static_cast<double>(RAND_MAX)));
            return weight();
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
