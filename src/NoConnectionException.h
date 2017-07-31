#ifndef WZANN_NOCONNECTIONEXCEPTION_H_
#define WZANN_NOCONNECTIONEXCEPTION_H_


#include <stdexcept>


namespace wzann {
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
} // namespace wzann

#endif // NOCONNECTIONEXCEPTION_H
