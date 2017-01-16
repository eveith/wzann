#ifndef ELMANNETWORKPATTERNTEST_H
#define ELMANNETWORKPATTERNTEST_H


#include <gtest/gtest.h>

#include "ActivationFunction.h"


class ElmanNetworkPatternTest: public ::testing::Test
{

private:
    QList<int> layers;
    QList<Winzent::ANN::ActivationFunction*>
        activationFunctions;

protected:
    ElmanNetworkPatternTest();
};


#endif // ELMANNETWORKPATTERNTEST_H
