#include <gtest/gtest.h>

#include "ActivationFunctionTest.h"

#include "SigmoidActivationFunction.h"
#include "LinearActivationFunction.h"



using namespace Winzent::ANN;


ActivationFunctionTest::ActivationFunctionTest(QObject *parent):
        QObject(parent)
{
}


TEST(ActivationFunctionTest, testSigmoidActivationFunction)
{
    SigmoidActivationFunction a;

    ASSERT_TRUE(a.hasDerivative());
    ASSERT_EQ(1 / (1 + std::exp(-8.0, a.calculate(8.0))));
    ASSERT_EQ(1.5, 1.0 + a.calculate(0.0));
    ASSERT_EQ(0.5 * (1.0 - 0.5), a.calculateDerivative(5.0, 0.5));
}


TEST(ActivationFunctionTest, testLinarActivationFunction)
{
    LinearActivationFunction a;

    ASSERT_TRUE(a.hasDerivative());
    ASSERT_EQ(1.5, a.calculate(1.5));
    ASSERT_EQ(-2.75, a.calculate(-2.75));

    LinearActivationFunction b(2.0);
    ASSERT_EQ(3.0, b.calculate(1.5));
    ASSERT_EQ(-5.5, b.calculate(-2.75));
}
