#ifndef REVOLUTIONARYTRAININGALGORITHMTEST_H
#define REVOLUTIONARYTRAININGALGORITHMTEST_H


namespace Winzent {
    namespace ANN {
        class NeuralNetwork;
    }
}


using wzann::NeuralNetwork;
using wzann::Individual;


class REvolutionaryTrainingAlgorithmTest: public ::testing::Test
{
private:
    NeuralNetwork* createNeuralNetwork();
};


#endif // REVOLUTIONARYTRAININGALGORITHMTEST_H
