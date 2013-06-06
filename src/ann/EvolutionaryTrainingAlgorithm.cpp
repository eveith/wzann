#include <limits>
#include <cstdlib>

#include <QObject>
#include <QList>

#include "NeuralNetwork.h"
#include "Connection.h"

#include "EvolutionaryTrainingAlgorithm.h"




namespace Winzent {
    namespace ANN {


        double EvolutionaryTrainingAlgorithm::Chromosome
                ::defaultMutationAmplitude = 5.0;


        EvolutionaryTrainingAlgorithm::Chromosome::Chromosome(
                const NeuralNetwork *&network,
                QObject *parent):
                    QObject(parent),
                    m_genes(QList<Connection *>()),
                    m_network(network->clone()),
                    m_mutationAmplitude(defaultMutationAmplitude)
        {
        }


        qreal EvolutionaryTrainingAlgorithm::Chromosome::mutationAmplitude()
                const
        {
            return m_mutationAmplitude;
        }


        EvolutionaryTrainingAlgorithm::Chromosome
        *EvolutionaryTrainingAlgorithm::Chromosome
                ::mutationAmplitude(qreal amplitude)
        {
            m_mutationAmplitude = amplitude;
            return this;
        }


        EvolutionaryTrainingAlgorithm::EvolutionaryTrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent):
                    TrainingAlgorithm(network, parent)
        {
        }


        double EvolutionaryTrainingAlgorithm::mutate(const double &d) const
        {
#if 0
            return d
                    + (
                        m_mutateMin
                        + (m_mutateMax - m_mutateMin)
                        * double(qrand()) / RAND_MAX);
#endif
            return d;
        }


        void EvolutionaryTrainingAlgorithm::train(
                NeuralNetwork *network,
                TrainingSet *trainingSet)
        {
        }


        qreal EvolutionaryTrainingAlgorithm::dc1(
                qreal y,
                const qreal &u,
                const qreal &t)
        {
            if (t != 0) {
                y += (u - y) / t;
            } else {
                y = u;
            }

            return y;
        }
        

        qreal EvolutionaryTrainingAlgorithm::frandom()
        {
            return (qrand() % std::numeric_limits<int>::max())
                    / static_cast<qreal>(std::numeric_limits<int>::max() + 1);
        }
    } // namespace ANN
} // namespace Winzent
