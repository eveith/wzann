#include <gtest/gtest.h>

#include "ActivationFunction.h"
#include "ActivationFunctionTest.h"


using namespace wzann;


TEST(ActivationFunctionTest, testIdentity)
{
    ASSERT_EQ(
            1.5,
            calculate(ActivationFunction::Identity, 1.5));
    ASSERT_EQ(
            -2.75,
            calculate(ActivationFunction::Identity, -2.75));
    ASSERT_EQ(
            1,
            calculateDerivative(ActivationFunction::Identity, 3.2));
}


TEST(ActivationFunctionTest, testBinaryStep)
{
    ASSERT_EQ(
            1.,
            calculate(ActivationFunction::BinaryStep, 1.5));
    ASSERT_EQ(
            0.,
            calculate(ActivationFunction::BinaryStep, -2.75));
    ASSERT_EQ(
            0.,
            calculateDerivative(ActivationFunction::BinaryStep, 3.2));
}


TEST(ActivationFunctionTest, testLogistic)
{
    ASSERT_EQ(
            1.0 / (1.0 + std::exp(-1.5)),
            calculate(ActivationFunction::Logistic, 1.5));
    ASSERT_EQ(
            1.0 / (1.0 + std::exp(2.75)),
            calculate(ActivationFunction::Logistic, -2.75));
    ASSERT_EQ(
            calculate(ActivationFunction::Logistic, 3.2)
                * (1.0 - calculate(ActivationFunction::Logistic, 3.2)),
            calculateDerivative(ActivationFunction::Logistic, 3.2));
}


TEST(ActivationFunctionTest, testTanh)
{
    ASSERT_EQ(
            std::tanh(1.5),
            calculate(ActivationFunction::Tanh, 1.5));
    ASSERT_EQ(
            std::tanh(-2.75),
            calculate(ActivationFunction::Tanh, -2.75));
    ASSERT_EQ(
            1. - calculate(ActivationFunction::Tanh, 3.2)
                * calculate(ActivationFunction::Tanh, 3.2),
            calculateDerivative(ActivationFunction::Tanh, 3.2));
}
