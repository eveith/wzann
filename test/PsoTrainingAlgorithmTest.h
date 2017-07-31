#ifndef PSOTRAININGALGORITHMTEST_H
#define PSOTRAININGALGORITHMTEST_H


#include <gtest/gtest.h>


namespace Winzent {
    namespace ANN {
        class NeuralNetwork;
        class Individual;
    }
}


using wzann::NeuralNetwork;
using wzann::Individual;


class PsoTrainingAlgorithmTest: public ::testing::Test
{
private:
    NeuralNetwork* createNeuralNetwork();
};


#endif // PSOTRAININGALGORITHMTEST_H
