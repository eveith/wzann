#include <cmath>

#include "ActivationFunction.h"


namespace wzann {


    /*!
     * \brief Calculates the result of a pass of an activation function
     *
     * \param f The activation function to use
     *
     * \param x The input parameter x, as in f(x)
     *
     * \return The f(x)
     *
     * \sa ActivationFunction
     */
    double calculate(ActivationFunction f, double x)
    {
        double fx;

        switch (f) {
        case ActivationFunction::Null:
            fx = 0.;
            break;
        case ActivationFunction::Identity:
            fx = x;
            break;
        case ActivationFunction::BinaryStep:
            fx = x + 1. < 1. ? 0. : 1.;
            break;
        case ActivationFunction::Logistic:
            fx = 1. / (1. + std::exp(-x));
            break;
        case ActivationFunction::Tanh:
            fx = std::tanh(x);
            break;
        case ActivationFunction::ReLU:
            fx = x + 1. < 1. ? 0. : x;
            break;
        case ActivationFunction::Gaussian:
            fx = std::exp(- std::pow(x, 2));
            break;
        default:
            throw "Unknown activation function";
        }

        return fx;
    }


    /*!
     * \brief Calculates the result of f'(x).
     *
     * \param f The activation function
     *
     * \param x The input parameter
     *
     * \return f'(x)
     */
    double calculateDerivative(ActivationFunction f, double x)
    {
        double fx;

        switch (f) {
        case ActivationFunction::Null:
            fx = 1.;
            break;
        case ActivationFunction::Identity:
            fx = 1.;
            break;
        case ActivationFunction::BinaryStep:
            fx = 0.;
            break;
        case ActivationFunction::Logistic:
            fx = calculate(f, x) * (1. - calculate(f, x));
            break;
        case ActivationFunction::Tanh:
            fx = 1. - std::pow(calculate(f, x), 2);
            break;
        case ActivationFunction::ReLU:
            fx = x + 1. < 1. ? 0. : 1.;
            break;
        case ActivationFunction::Gaussian:
            fx = -2. * x * std::exp(- std::pow(x, 2));
            break;
        default:
            throw "Unknown activation function";
        }

        return fx;
    }
}
