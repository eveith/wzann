#ifndef WZANN_UNKNOWNNEURONEXCEPTION_H_
#define WZANN_UNKNOWNNEURONEXCEPTION_H_


#include <stdexcept>


namespace wzann {
    class Neuron;


    /*!
     * \brief The UnknownNeuronException class indicates that a neuron
     *  that should be used in the context of a NeuralNetwork is not part
     *  of that NeuralNetwork object.
     */
    class UnknownNeuronException : public std::invalid_argument
    {
    public:
        UnknownNeuronException(Neuron const& neuron):
                std::invalid_argument("Neuron unknown to NeuralNetwork"),
                m_neuron(&neuron)
        {
        }


        virtual ~UnknownNeuronException()
        {
        }


        //! \brief Returns the unknown neuron
        Neuron const& neuron() const
        {
            return *m_neuron;
        }


    private:


        Neuron const* m_neuron;
    };
} // namespace wzann

#endif // WZANN_UNKNOWNNEURONEXCEPTION_H_
