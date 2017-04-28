#ifndef NOCONNECTIONEXCEPTION_H
#define NOCONNECTIONEXCEPTION_H


#include <stdexcept>


namespace Winzent {
    namespace ANN {
        class Neuron;


        class NoConnectionException : public std::invalid_argument
        {
        public:
            NoConnectionException(Neuron const& from, Neuron const& to):
                    std::invalid_argument("No connection between "
                        "the two neurons"),
                    m_from(&from),
                    m_to(&to)
            {
            }


            virtual ~NoConnectionException()
            {
            }


            //! \brief The neuron from which the connection should have begun
            Neuron const& from() const
            {
                return *m_from;
            }


            //! \brief The neuron to which the connection should have lead
            Neuron const& to() const
            {
                return *m_to;
            }


        private:


            Neuron const* m_from;


            Neuron const* m_to;
        };

    } // namespace ANN
} // namespace Winzent

#endif // NOCONNECTIONEXCEPTION_H
