#ifndef REVOLUTIONARYTRAININGALGORITHMTEST_H
#define REVOLUTIONARYTRAININGALGORITHMTEST_H


#include <QObject>


namespace Winzent {
    namespace ANN {
        class NeuralNetwork;
    }
}


using Winzent::ANN::NeuralNetwork;


class REvolutionaryTrainingAlgorithmTest: public QObject
{
    Q_OBJECT

private:
    NeuralNetwork *createNeuralNetwork();

public:
    explicit REvolutionaryTrainingAlgorithmTest(QObject *parent = 0);

private slots:
    void testAgeIndividual();
    void testParametersSettingAndRetrieval();
};

#endif // REVOLUTIONARYTRAININGALGORITHMTEST_H
