#include <limits>
#include <cstdlib>
#include <cmath>

#include <QObject>
#include <QList>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Connection.h"

#include "TrainingSet.h"
#include "TrainingAlgorithm.h"

#include "REvolutionaryTrainingAlgorithm.h"


using std::exp;
using std::fabs;


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


        QList<qreal> &Individual::errorVector()
        {
            return m_errorVector;
        }


        const QList<qreal> &Individual::errorVector() const
        {
            return m_errorVector;
        }


        int Individual::compare(const Individual *other) const
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


        bool Individual::isBetterThan(const Individual *other) const
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


        REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent):
                    TrainingAlgorithm(network, parent),
                    m_populationSize(0),
                    m_eliteSize(0),
                    m_gradientWeight(1.8),
                    m_errorWeight(1.0),
                    m_eamin(1e-30),
                    m_ebmin(1e-7),
                    m_ebmax(1e-1)
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


        int REvolutionaryTrainingAlgorithm::eliteSize() const
        {
            return m_eliteSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::eliteSize(const int &size)
        {
            m_eliteSize = size;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::gradientWeight() const
        {
            return m_gradientWeight;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::gradientWeight(const qreal &weight)
        {
            m_gradientWeight = weight;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::errorWeight() const
        {
            return m_errorWeight;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::errorWeight(const qreal &weight)
        {
            m_errorWeight = weight;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::eamin() const
        {
            return m_eamin;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::eamin(const qreal &eamin)
        {
            m_eamin = eamin;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::ebmin() const
        {
            return m_ebmin;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::ebmin(const qreal &ebmin)
        {
            m_ebmin = ebmin;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::ebmax() const
        {
            return m_ebmax;
        }

        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::ebmax(const qreal &ebmax)
        {
            m_ebmax = ebmax;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::applyDxBounds(
                const qreal &dx,
                const qreal &parameter)
                const
        {
            qreal cdx = dx;

            if (dx < ebmin() * fabs(parameter)) {
                cdx = ebmin() * fabs(parameter);
            } else if (dx > ebmax() * fabs(parameter)) {
                cdx = ebmax() * fabs(parameter);
            } else if (dx < eamin()) {
                cdx = eamin();
            }

            return cdx;
        }


        Individual *REvolutionaryTrainingAlgorithm::generateIndividual(
                const QList<Individual *> &population,
                TrainingSet *const &trainingSet)
        {
            Individual *newIndividual = new Individual(
                        population.first()->neuralNetwork());
            Individual *eliteIndividual = population.at(std::abs(
                    qrand() % eliteSize() - qrand() % eliteSize()));
            Individual *otherIndividual = population.at(
                        qrand() % population.size());

            if (otherIndividual->isBetterThan(eliteIndividual)) {
                Individual *tmp = eliteIndividual;
                eliteIndividual = otherIndividual;
                otherIndividual = tmp;
            }

            qreal xlp = 0.0;
            qreal errorRate = trainingSet->error() / trainingSet->targetError()
                    -1.0;
            int gradientSwitch = qrand() % 3;
            qreal expvar = exp(frandom() - frandom());

            if (2 == gradientSwitch) {
                xlp = (frandom() + frandom() + frandom() + frandom() + frandom()
                        + frandom() + frandom() + frandom() + frandom()
                        + frandom() - frandom() - frandom() - frandom()
                        - frandom() - frandom() - frandom()) * gradientWeight();

                if (xlp > 0.0) {
                    xlp *= 0.5;
                }

                xlp *= exp(errorRate * gradientWeight());
            }

            // Now modify the new individual:

            for (int i = 0; i != newIndividual->parameters().size(); ++i) {
                qreal dx = eliteIndividual->scatter().at(i) * exp(
                        errorWeight() *
                            (trainingSet->error() -trainingSet->targetError()));

                dx = applyDxBounds(dx, eliteIndividual->parameters().at(i));

                eliteIndividual->scatter()[i] = dx;

                if (frandom() >= 0.5) {
                    dx = 0.5 * (eliteIndividual->scatter().at(i)
                            + otherIndividual->scatter().at(i));
                }

                dx *= expvar;
                dx = applyDxBounds(dx, eliteIndividual->parameters().at(i));

                newIndividual->scatter()[i] = dx;

                dx = newIndividual->scatter().at(i) * (frandom() + frandom()
                        + frandom() + frandom() + frandom() - frandom()
                        - frandom() - frandom() - frandom() - frandom());

                if (0 == gradientSwitch) { // Everything from the elite, p=2/3
                    if (frandom() < 2.0/3.0) {
                        dx += eliteIndividual->parameters().at(i);
                    } else {
                        dx += otherIndividual->parameters().at(i);
                    }
                } else if (1 == gradientSwitch) { // use eliteIndividual
                    dx += eliteIndividual->parameters().at(i);
                } else if (2 == gradientSwitch) { // use elite & gradient
                    dx += eliteIndividual->parameters().at(i);
                    dx += xlp * (eliteIndividual->parameters().at(i)
                            - otherIndividual->parameters().at(i));
                }

                newIndividual->parameters()[i] = dx;
            }

            return newIndividual;
        }
    } // namespace ANN
} // namespace Winzent
