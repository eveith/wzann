#ifndef WEIGHTFIXEDEXCEPTION_H
#define WEIGHTFIXEDEXCEPTION_H


#include <stdexcept>


namespace Winzent {
    namespace ANN {


        /*!
         * \brief The WeightFixedException class indicates a logic error and
         *  is thrown when the weight of a connection whose weight value is
         *  fixed should be changed.
         */
        class WeightFixedException: public std::logic_error
        {
        public:


            WeightFixedException();

            virtual ~WeightFixedException();
        };

    } // namespace ANN
} // namespace Winzent

#endif // WEIGHTFIXEDEXCEPTION_H
