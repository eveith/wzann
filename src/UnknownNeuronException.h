#ifndef UNKNOWNNEURONEXCEPTION_H
#define UNKNOWNNEURONEXCEPTION_H


#include <stdexcept>


namespace Winzent {
    namespace ANN {
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
    } // namespace ANN
} // namespace Winzent

#endif // UNKNOWNNEURONEXCEPTION_H
