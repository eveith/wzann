#ifndef WZANN_LAYERSIZEMISMATCHEXCEPTION_H_
#define WZANN_LAYERSIZEMISMATCHEXCEPTION_H_


#include <cstdlib>
#include <stdexcept>


namespace wzann {


    /*!
     * \brief The LayerSizeMismatchException class is thrown whenever a
     *  method expects a Vector of a certain size, but receives a Vector
     *  of a different size.
     */
    class LayerSizeMismatchException : public std::invalid_argument
    {
    public:


        /*!
         * \brief Constructs a new exception
         *
         * \param[in] expected The expected size of the Vector
         *
         * \param[in] actual The actual size of the supplied Vector
         */
        LayerSizeMismatchException(size_t expected, size_t actual):
                std::invalid_argument("Supplied Vector argument has "
                    " the wrong number of elements"),
                m_expected(expected),
                m_actual(actual)
        {
        }


        size_t expected() const
        {
            return m_expected;
        }


        size_t actual() const
        {
            return m_actual;
        }


    private:


        size_t m_expected;


        size_t m_actual;
    };
} // namespace wzann

#endif // WZANN_LAYERSIZEMISMATCHEXCEPTION_H_
