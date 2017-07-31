#ifndef WZANN_WEIGHTFIXEDEXCEPTION_H_
#define WZANN_WEIGHTFIXEDEXCEPTION_H_


#include <stdexcept>


namespace wzann {


    /*!
     * \brief The WeightFixedException class indicates a logic error and
     *  is thrown when the weight of a connection whose weight value is
     *  fixed should be changed.
     */
    class WeightFixedException : public std::logic_error
    {
    public:


        WeightFixedException();

        virtual ~WeightFixedException();
    };
} // namespace wzann

#endif // WZANN_WEIGHTFIXEDEXCEPTION_H_
