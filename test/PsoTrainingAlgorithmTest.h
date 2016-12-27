#ifndef PSOTRAININGALGORITHMTEST_H
#define PSOTRAININGALGORITHMTEST_H


#include <QObject>
#include <QList>


namespace Winzent {
    namespace ANN {
        class NeuralNetwork;
        class Individual;
    }
}


using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::Individual;


class PsoTrainingAlgorithmTest: public QObject
{
    Q_OBJECT

private:
    NeuralNetwork *createNeuralNetwork();

public:
    explicit PsoTrainingAlgorithmTest(QObject *parent = 0);

private slots:
    void testTrainXOR();};

#endif // PSOTRAININGALGORITHMTEST_H
