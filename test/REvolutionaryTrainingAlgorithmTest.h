#ifndef REVOLUTIONARYTRAININGALGORITHMTEST_H
#define REVOLUTIONARYTRAININGALGORITHMTEST_H


namespace Winzent {
    namespace ANN {
        class NeuralNetwork;
    }
}


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Individual;


class REvolutionaryTrainingAlgorithmTest: public ::testing::Test
{
private:
    NeuralNetwork* createNeuralNetwork();
};


#endif // REVOLUTIONARYTRAININGALGORITHMTEST_H
