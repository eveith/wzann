#ifndef WZANN_ACTIVATIONFUNCTION_H_
#define WZANN_ACTIVATIONFUNCTION_H_


#include <cstdint>

#include "enum.h"
#include "LibVariantSupport.h"


namespace wzann {


    /*!
     * \brief Activation functions for neurons in an ANN
     *
     * This enum discriminates the different activation functions, given
     * as parameter to the ::calculate() and ::calculateDerivative
     * functions.
     *
     * The different activation functions available are:
     *
     *  * Identity
     *  * Binary Step
     *  * Logistic (alias Sigmoid)
     *  * Tanh
     *  * ReLU
     *  * Gaussian
     *
     * \sa <https://en.wikipedia.org/wiki/Activation_function>
     */
    BETTER_ENUM(ActivationFunction, intmax_t,
            Null,
            Identity,
            BinaryStep,
            Logistic,
            Tanh,
            ReLU,
            Gaussian)

    /*!
     * \brief Calculates $f(x)$
     *
     * \param f The function
     *
     * \param x The argument
     *
     * \return $f(x)$
     *
     * \sa ActivationFunction
     */
    double calculate(ActivationFunction f, double x);


    /*!
     * \brief Calculates $f'(x)$
     *
     * \param f The function
     *
     * \param x The argument
     *
     * \return $f'(x)$
     *
     * \sa ActivationFunction
     */
    double calculateDerivative(ActivationFunction f, double x);


    template <>
    inline libvariant::Variant to_variant(ActivationFunction const& af)
    {
        return libvariant::Variant(af._to_string());
    }


    template <>
    inline ActivationFunction from_variant(libvariant::Variant const& v)
    {
        return ActivationFunction::_from_string(v.AsString().c_str());
    }


    template <>
    inline ActivationFunction* new_from_variant(
            libvariant::Variant const& v)
    {
        auto af = from_variant<ActivationFunction>(v);
        return new ActivationFunction(af);
    }
} // namespace wzann

#endif /* WZANN_ACTIVATIONFUNCTION_H_ */
