#ifndef REVOLUTIONARYTRAININGALGORITHMTEST_H_
#define REVOLUTIONARYTRAININGALGORITHMTEST_H_


namespace wzann {
    class NeuralNetwork;
}


using wzann::NeuralNetwork;


class REvolutionaryTrainingAlgorithmTest: public ::testing::Test
{
public:
    NeuralNetwork* createNeuralNetwork();
};


#endif // REVOLUTIONARYTRAININGALGORITHMTEST_H
