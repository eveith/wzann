#ifndef UNKNOWNNEURONEXCEPTION_H
#define UNKNOWNNEURONEXCEPTION_H


namespace Winzent {
    namespace ANN {

        class UnknownNeuronException : public std::invalid_argument
        {
        public:
            UnknownNeuronException();
        };

    } // namespace ANN
} // namespace Winzent

#endif // UNKNOWNNEURONEXCEPTION_H