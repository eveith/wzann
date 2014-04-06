#include <limits>
#include <cstdlib>

#include <QObject>
#include <QList>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Connection.h"

#include "REvolutionaryTrainingAlgorithm.h"




namespace Winzent {
    namespace ANN {


        Individual::Individual(NeuralNetwork *neuralNetwork):
                m_neuralNetwork(neuralNetwork),
                m_timeToLive(0)
        {
        }


        Individual::~Individual()
        {
            delete m_neuralNetwork;
        }


        QList<qreal> Individual::scatter() const
        {
            return m_scatter;
        }


        Individual &Individual::scatter(QList<qreal> scatter)
        {
            m_scatter = scatter;
            return *this;
        }


        QList<qreal> Individual::parameters() const
        {
            QList<qreal> r;

            for (int i = 0; i != m_neuralNetwork->size(); ++i) {
                Layer *layer = m_neuralNetwork->layerAt(i);

                for (int j = 0; j != layer->size(); ++j) {
                    Neuron *neuron = layer->neuronAt(j);
                    QList<Connection *> connections =
                            m_neuralNetwork->neuronConnectionsFrom(neuron);
                    foreach (Connection *connection, connections) {
                        if (!connection->fixedWeight()) {
                            r << connection->weight();
                        }
                    }
                }
            }

            return r;
        }


        Individual &Individual::parameters(QList<qreal> parameters)
        {
            int w = 0;
            for (int i = 0; i != m_neuralNetwork->size(); ++i) {
                Layer *layer = m_neuralNetwork->layerAt(i);

                for (int j = 0; j != layer->size(); ++j) {
                    Neuron *neuron = layer->neuronAt(j);
                    QList<Connection *> connections =
                            m_neuralNetwork->neuronConnectionsFrom(neuron);
                    foreach (Connection *connection, connections) {
                        if (!connection->fixedWeight()) {
                            connection->weight(parameters[w]);
                            w++;
                        }
                    }
                }
            }

            return *this;
        }


        int Individual::timeToLive() const
        {
            return m_timeToLive;
        }


        Individual &Individual::timeToLive(const int &ttl)
        {
            m_timeToLive = ttl;
            return *this;
        }


        QList<qreal> Individual::errorVector()
        {
            return m_errorVector;
        }


        const QList<qreal> Individual::errorVector() const
        {
            return m_errorVector;
        }


        int Individual::compare(const Individual *&other)
        {
            if (this->timeToLive() < 0 && other->timeToLive() >= 0) {
                return -1;
            }

            if (this->timeToLive() >= 0 && other->timeToLive() < 0) {
                return 1;
            }

            if (this->errorVector().size() > 1
                    && other->errorVector().size() > 1) {
                for (int i = 1; i != this->errorVector().size(); ++i) {
                    if (this->errorVector().at(i) < other->errorVector().at(i)){
                        return 1;
                    } else if (other->errorVector().at(i)
                            < this->errorVector().at(i)) {
                        return -1;
                    }
                }
            }

            if (this->errorVector().at(0) == other->errorVector().at(0)) {
                if (this->timeToLive() == other->timeToLive()) {
                    return 0;
                } else if (this->timeToLive() > other->timeToLive()) {
                    return 1;
                } else {
                    return -1;
                }
            } else if (this->errorVector().at(0) < other->errorVector().at(0)) {
                return 1;
            } else {
                return -1;
            }

            return 0; // only fall-through, won't be reached
        }


        bool Individual::isBetterThan(const Individual *&other)
        {
            return (1 == compare(other));
        }


        qreal REvolutionaryTrainingAlgorithm::dc1(
                const qreal &y,
                const qreal &u,
                const qreal &t)
        {
            qreal r = 0.0;

            if (t != 0) {
                r = y + (u - y) / t;
            } else {
                r = u;
            }

            return r;
        }
        

        qreal REvolutionaryTrainingAlgorithm::frandom()
        {
            return ((qrand() % RAND_MAX) / static_cast<qreal>(RAND_MAX));
        }


        REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm():
                TrainingAlgorithm(network, parent),
                m_populationSize(0),
                m_eliteSize(0)
        {
        }


        int REvolutionaryTrainingAlgorithm::populationSize() const
        {
            return m_populationSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::populationSize(const int &size)
        {
            m_populationSize = size;
            return *this;
        }


        int REvolutionaryTrainingAlgorithm::eliteSize()
        {
            return m_eliteSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::eliteSize(const int &size)
        {
            m_eliteSize = size;
            return *this;
        }


        NeuralNetwork *generateIndividual(
                const QList<Individual *> &population)
        {
            Individual *newIndividual = new Individual(
                        population.first()->neuralNetwork);
            Individual *eliteIndividual = population.at(std::abs(
                    qrand() % eliteSize() - qrand() % eliteSize()));
            Individual *otherIndividual = population.at(
                        qrand() % population.size());

            if (otherIndividual->isBetterThan(eliteIndividual)) {
                Individual *tmp = eliteIndividual;
                eliteIndividual = otherIndividual;
                otherIndiviual  = eliteIndividual;
            }

            qreal errorRate = error() / targetError() - 1.0;

            return newIndividual;
        }
    } // namespace ANN
} // namespace Winzent
