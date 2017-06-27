#ifndef ELMANNETWORKPATTERNTEST_H
#define ELMANNETWORKPATTERNTEST_H


#include <gtest/gtest.h>

#include "NeuralNetworkPattern.h"


class ElmanNetworkPatternTest: public ::testing::Test
{

protected:
    Winzent::ANN::NeuralNetworkPattern::SimpleLayerDefinitions m_layers;

protected:
    ElmanNetworkPatternTest();
};


#endif // ELMANNETWORKPATTERNTEST_H
