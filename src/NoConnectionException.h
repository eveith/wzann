#ifndef NOCONNECTIONEXCEPTION_H
#define NOCONNECTIONEXCEPTION_H


namespace Winzent {
    namespace ANN {

        class NoConnectionException : public std::invalid_argument
        {
        public:
            NoConnectionException();
        };

    } // namespace ANN
} // namespace Winzent

#endif // NOCONNECTIONEXCEPTION_H