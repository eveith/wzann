#ifndef REVOLUTIONARYTRAININGALGORITHMTEST_H
#define REVOLUTIONARYTRAININGALGORITHMTEST_H


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


class REvolutionaryTrainingAlgorithmTest: public QObject
{
    Q_OBJECT

private:
    NeuralNetwork *createNeuralNetwork();

public:
    explicit REvolutionaryTrainingAlgorithmTest(QObject *parent = 0);

private slots:
    void testIndividualInitialization();
    void testAgeIndividual();
    void testIndividualOperatorEquals();
    void testIndividualOperatorAssign();
    void testParametersSettingAndRetrieval();
    void testCompareIndividuals();
    void testModifyIndividual();
    void testSortPopulation();
    void testTrainXOR();

    void recordIteration(
            const int &epoch,
            const qreal &error,
            const QList<Individual *> &population);
};

#endif // REVOLUTIONARYTRAININGALGORITHMTEST_H
