#include <limits>
#include <cfenv>
#include <cmath>
#include <cstdlib>

#include <QtDebug>

#include <QObject>
#include <QList>

#include <log4cxx/logger.h>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Connection.h"

#include "TrainingSet.h"
#include "TrainingAlgorithm.h"

#include "REvolutionaryTrainingAlgorithm.h"


#pragma STDC FENV_ACCESS ON


using std::exp;
using std::fabs;


namespace Winzent {
    namespace ANN {


        Individual::Individual(NeuralNetwork *neuralNetwork):
                m_neuralNetwork(neuralNetwork),
                m_timeToLive(0)
        {
            m_errorVector << std::numeric_limits<qreal>::infinity();

            neuralNetwork->eachConnection([this](const Connection *const &c) {
                if (!c->fixedWeight()) {
                    m_scatter << 0.2;
                }
            });
        }


        Individual::~Individual()
        {
            delete m_neuralNetwork;
        }


        NeuralNetwork *const &Individual::neuralNetwork()
        {
            return m_neuralNetwork;
        }


        const NeuralNetwork *Individual::neuralNetwork() const
        {
            return m_neuralNetwork;
        }


        NeuralNetwork *Individual::neuralNetworkClone() const
        {
            return m_neuralNetwork->clone();
        }


        ValueVector Individual::scatter() const
        {
            return m_scatter;
        }


        ValueVector &Individual::scatter()
        {
            return m_scatter;
        }


        Individual &Individual::scatter(ValueVector scatter)
        {
            m_scatter = scatter;
            return *this;
        }


        ValueVector Individual::parameters() const
        {
            ValueVector r;

            neuralNetwork()->eachConnection([&r](const Connection *const &c) {
                if (!c->fixedWeight()) {
                    r << c->weight();
                }
            });

            return r;
        }


        Individual &Individual::parameters(ValueVector parameters)
        {
            int i = 0;
            neuralNetwork()->eachConnection([&parameters, &i](
                    Connection *const &c) {
                if (!c->fixedWeight()) {
                    c->weight(parameters.at(i++));
                }
            });

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


        Individual &Individual::age()
        {
            timeToLive(timeToLive() - 1);
            return *this;
        }


        bool Individual::isAlive() const
        {
            return (timeToLive() > 0);
        }

        ValueVector &Individual::errorVector()
        {
            return m_errorVector;
        }


        const ValueVector &Individual::errorVector() const
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

            if (this->errorVector().at(0) < other->errorVector().at(0)) {
                return 1;
            } else if (this->errorVector().at(0) > other->errorVector().at(0)) {
                return -1;
            } else {
                int size = (
                        this->errorVector().size() > other->errorVector().size()
                        ? other->errorVector().size()
                        : this->errorVector().size());
                for (int i = 1; i < size; ++i) {
                    if (this->errorVector().at(i) < other->errorVector().at(i)){
                        return 1;
                    } else if (other->errorVector().at(i)
                            < this->errorVector().at(i)) {
                        return -1;
                    }
                }
            }

            return 0;
        }


        bool Individual::isBetterThan(const Individual *other) const
        {
            return (1 == compare(other));
        }


        bool Individual::isIndividual1Better(
                const Individual *const &i1,
                const Individual *const &i2)
        {
            return i1->isBetterThan(i2);
        }


        bool Individual::operator ==(const Individual &other) const
        {
#ifdef      QT_DEBUG
                bool ok = true;
                ok &= timeToLive() == other.timeToLive();
                ok &= other.errorVector() == errorVector();
                ok &= other.scatter() == scatter();
                ok &= other.parameters() == parameters();
                return ok;
#else
                return (other.timeToLive() == timeToLive()
                        && other.errorVector() == errorVector()
                        && other.scatter() == scatter()
                        && other.parameters() == parameters());
#endif
        }


        Individual &Individual::operator =(const Individual &other)
        {
            if (this == &other) {
                return *this;
            }

            m_timeToLive    = other.m_timeToLive;
            m_errorVector   = other.m_errorVector;
            m_scatter       = other.m_scatter;

            if (nullptr != m_neuralNetwork) {
                delete m_neuralNetwork;
            }

            m_neuralNetwork = other.neuralNetworkClone();

            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::dc1(
                const qreal &y,
                const qreal &u,
                const qreal &t)
        {
            qreal r = 0.0;

            if (t != 0) {
                r = y + ((u - y) / t);
            } else {
                r = u;
            }

            return r;
        }


        void REvolutionaryTrainingAlgorithm::sortPopulation(
                QList<Individual *> &population)
        {
            qSort(
                    population.begin(),
                    population.end(),
                    &Individual::isIndividual1Better);
        }


        qreal REvolutionaryTrainingAlgorithm::frandom()
        {
            return ((qrand() % RAND_MAX - 1) / static_cast<qreal>(RAND_MAX));
        }


        REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent):
                    TrainingAlgorithm(network, parent),
                    m_maxNoSuccessEpochs(0),
                    m_populationSize(0),
                    m_eliteSize(0),
                    m_gradientWeight(1.8),
                    m_successWeight(1.0),
                    m_eamin(1e-30),
                    m_ebmin(1e-7),
                    m_ebmax(1e-1),
                    m_startTTL(0),
                    m_measurementEpochs(5000),
                    m_success(0.25),
                    m_targetSuccess(0.25)
        {
        }


        int REvolutionaryTrainingAlgorithm::maxNoSuccessEpochs() const
        {
            return m_maxNoSuccessEpochs;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::maxNoSuccessEpochs(const int &epochs)
        {
            m_maxNoSuccessEpochs = epochs;
            return *this;
        }


        int REvolutionaryTrainingAlgorithm::populationSize() const
        {
            return m_populationSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::populationSize(const int &size)
        {
            m_populationSize = size;

            if (0 == eliteSize()) {
                eliteSize(std::ceil(size * 0.1));
            }

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


        qreal REvolutionaryTrainingAlgorithm::successWeight() const
        {
            return m_successWeight;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::successWeight(const qreal &weight)
        {
            m_successWeight = weight;
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


        int REvolutionaryTrainingAlgorithm::startTTL() const
        {
            return m_startTTL;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::startTTL(const int &ttl)
        {
            m_startTTL = ttl;
            return *this;
        }


        int REvolutionaryTrainingAlgorithm::measurementEpochs() const
        {
            return m_measurementEpochs;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::measurementEpochs(const int &epochs)
        {
            m_measurementEpochs = epochs;
            return *this;
        }


        qreal REvolutionaryTrainingAlgorithm::applyDxBounds(
                const qreal &dx,
                const qreal &parameter)
                const
        {
            qreal cdx = dx;

            if (std::fetestexcept(FE_UNDERFLOW)) {
                LOG4CXX_DEBUG(logger, "Underflow detected");
                cdx = eamin();
            }

            if (std::fetestexcept(FE_OVERFLOW)) {
                LOG4CXX_DEBUG(logger, "Overflow detected");
                cdx = ebmax() * fabs(parameter);
            }

            if (cdx < ebmin() * fabs(parameter)) {
                cdx = ebmin() * fabs(parameter);
            }

            if (cdx > ebmax() * fabs(parameter)) {
                cdx = ebmax() * fabs(parameter);
            }

            if (cdx < eamin()) {
                cdx = eamin();
            }

            std::feclearexcept(FE_ALL_EXCEPT);
            return cdx;
        }


        bool REvolutionaryTrainingAlgorithm::hasSensibleTrainingParameters()
                const
        {
            bool ok = true;

            if (0 >= populationSize()) {
                LOG4CXX_ERROR(logger, "Population size is 0");
                ok = false;
            }

            if (0 >= eliteSize()) {
                LOG4CXX_ERROR(logger, "Elite size is 0");
                ok = false;
            }

            if (eliteSize() >= populationSize()) {
                LOG4CXX_ERROR(logger, "Elite is bigger or equal to population");
                ok = false;
            }

            if (startTTL() <= 0) {
                LOG4CXX_ERROR(logger, "No sensible start TTL (<= 0)");
                ok = false;
            }

            if (0 >= measurementEpochs()) {
                LOG4CXX_ERROR(
                        logger,
                        "Invalid number of epochs for measurement,"
                            "must be > 0");
                ok = false;
            }

            return ok;
        }


        QList<Individual *>
        REvolutionaryTrainingAlgorithm::generateInitialPopulation(
                const NeuralNetwork * const &baseNetwork)
        {
            Individual *baseIndividual = new Individual(
                        baseNetwork->clone());
            QList <Individual *> population = { baseIndividual };

            for (int i = 1; i < populationSize() + 1; ++i) {
                Individual *individual = new Individual(baseNetwork->clone());
                ValueVector individualParameters = individual->parameters();

                for (int j = 0; j != individualParameters.size(); ++j) {
                    qreal r = baseIndividual->scatter().at(j) * exp(
                            0.4 * (0.5 - frandom()));
                    individual->scatter()[j] = r;
                    individualParameters[j] = baseIndividual->parameters().at(j)
                            + r * (frandom() - frandom()
                                + frandom() - frandom());
                }

                individual->parameters(individualParameters);
                individual->timeToLive(startTTL());

                population.append(individual);

                LOG4CXX_DEBUG(
                        logger,
                        "Created " << *individual);
            }

            Q_ASSERT(population.size() == populationSize() + 1);
            return population;
        }


        Individual *REvolutionaryTrainingAlgorithm::modifyIndividual(
                Individual *const &individual,
                QList<Individual *> &population)
        {
            bool populationContainedIndividual = false;

            if (population.contains(individual)) {
                population.removeAll(individual);
                populationContainedIndividual = true;
            }

            Individual *eliteIndividual = population.at(abs(
                    (qrand() % eliteSize()) - (qrand() % eliteSize())));
            Individual *otherIndividual = population.at(
                        qrand() % population.size());

            if (otherIndividual->isBetterThan(eliteIndividual)) {
                Individual *tmp = eliteIndividual;
                eliteIndividual = otherIndividual;
                otherIndividual = tmp;
            }

            qreal xlp = 0.0;
            qreal successRate = m_success / m_targetSuccess - 1.0;
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

                xlp *= exp(successWeight() * successRate);
            }

            // Now modify the new individual:

            int numParameters = eliteIndividual->parameters().size();
            ValueVector newParameters;
            newParameters.reserve(numParameters);

            Q_ASSERT(numParameters == eliteIndividual->parameters().size());
            Q_ASSERT(numParameters == otherIndividual->parameters().size());

            for (int i = 0; i != numParameters; ++i) {
                std::feclearexcept(FE_ALL_EXCEPT);

                qreal dx = eliteIndividual->scatter().at(i) * exp(
                        successWeight() * successRate);

                dx = applyDxBounds(dx, eliteIndividual->parameters().at(i));

                // Mutate scatter:

                eliteIndividual->scatter()[i] = dx;

                if (frandom() < 0.5) {
                    dx = eliteIndividual->scatter().at(i);
                } else {
                    dx = 0.5 * (eliteIndividual->scatter().at(i)
                            + otherIndividual->scatter().at(i));
                }

                dx *= expvar;
                dx = applyDxBounds(dx, eliteIndividual->parameters().at(i));

                // Generate new scatter:

                individual->scatter()[i] = dx;

                dx = dx * (frandom() + frandom()
                        + frandom() + frandom() + frandom() - frandom()
                        - frandom() - frandom() - frandom() - frandom());

                if (0 == gradientSwitch) { // Everything from the elite, p=2/3
                    if (qrand() % 3 < 2) {
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

                newParameters << dx;
            }

            Q_ASSERT(newParameters.size()
                     == individual->parameters().size());
            individual->parameters(newParameters);
            individual->timeToLive(startTTL());
            individual->errorVector().clear();
            individual->errorVector().append(
                    std::numeric_limits<qreal>::infinity());

#ifdef      QT_DEBUG
                for (int i = 0; i != newParameters.size(); ++i) {
                    Q_ASSERT(newParameters.at(i)
                             == individual->parameters().at(i));
                }
#endif

            LOG4CXX_DEBUG(logger, "Created " << *individual);

            if (populationContainedIndividual) {
                population.append(individual);
            }

            return individual;
        }


        void REvolutionaryTrainingAlgorithm::train(
                TrainingSet *const &trainingSet)
        {
            if (0 == startTTL()) {
                startTTL(std::ceil(trainingSet->maxEpochs() * 0.1));
            }

            if (0 == measurementEpochs()) {
                measurementEpochs(std::ceil(trainingSet->maxEpochs() / 200.0));
            }

            if (!hasSensibleTrainingParameters()) {
                LOG4CXX_ERROR(
                        logger,
                        "Training parameters have no sensible values, "
                            "won't train.");
                return;
            }

            int lastSuccess = 0;
            int epoch       = 0;
            QList<Individual *> population = generateInitialPopulation(
                    network());
            Individual *bestIndividual = population.first();

            do {
                // Modify the worst individual:

                if (0 != epoch) {
                    modifyIndividual(population.last(), population);
                }

                // Run current patterns through all networks
                // and age individuals:

                foreach (Individual *individual, population) {
                    int errorPos    = 1;
                    qreal totalMSE  = 0.0;

                    individual->errorVector().resize(
                            1 + trainingSet->trainingData().size());

                    foreach (TrainingItem item, trainingSet->trainingData()) {
                        ValueVector output = individual->neuralNetwork()
                                ->calculate(item.input());
                        qreal sampleMSE = calculateMeanSquaredError(
                                output,
                                item.expectedOutput());
 
                        individual->errorVector()[errorPos++] = sampleMSE;
                        totalMSE += sampleMSE;
                    }

                    individual->errorVector()[0] = totalMSE / errorPos;
                    individual->age();
                }

                // Check for addition of a new individual:

                Individual *newIndividual = population.last();
                Individual *worstIndividual = population.at(
                        population.size() - 2);

                // Check for global or, at least, local improvement:

                if (newIndividual->isBetterThan(bestIndividual)) {
                    lastSuccess = epoch;
                    bestIndividual = newIndividual;
                    bestIndividual->timeToLive(epoch);
                } else {
                    if (newIndividual->isBetterThan(worstIndividual)) {
                        if (worstIndividual->timeToLive() >= 0) {
                            m_success = dc1(
                                    m_success,
                                    1.0,
                                    measurementEpochs());
                        } else {
                            m_success = dc1(
                                    m_success,
                                    -1.0,
                                    measurementEpochs());
                        }
                    }
                }

                // Sort the list and do a bit of caretaking:

                sortPopulation(population);
                m_success = dc1(m_success, 0.0, measurementEpochs());
                epoch++;

                LOG4CXX_DEBUG(
                        logger,
                        "Iteration(epoch = " << epoch
                            << ", success = " << m_success
                            << ", targetSuccess = " << m_targetSuccess
                            << ", bestIndividual = " << *bestIndividual << ")");
            } while (population.first()->errorVector().first()
                        > trainingSet->targetError()
                    && epoch < trainingSet->maxEpochs()
                    && epoch - lastSuccess < maxNoSuccessEpochs());

            setFinalNumEpochs(*trainingSet, epoch);

            LOG4CXX_DEBUG(
                    logger,
                    "Training ended after " << epoch << " epochs; "
                        << *(population.first()));

            ValueVector bestParameters = bestIndividual->parameters();
            int i = 0;
            NeuralNetwork *trainedNetwork = network();

            trainedNetwork->eachConnection(
                    [&i, &bestParameters, &trainedNetwork](
                    Connection *const &c) {
                if (!c->fixedWeight()) {
                    c->weight(bestParameters.at(i++));
                }
            });

            setFinalError(
                    *trainingSet,
                    bestIndividual->errorVector().at(0));

            LOG4CXX_DEBUG(logger, population)

            // Cleanup:

            foreach (Individual *i, population) {
                delete i;
            }
        }
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator<<(ostream &os, const Winzent::ANN::Individual &individual)
    {
        os << "Individual(";

        os << "TTL = " << individual.timeToLive() << ", ";

        os << "Parameters = (";
        for (int i = 0; i < individual.parameters().size(); ++i) {
            os << individual.parameters().at(i);
            if (i < individual.parameters().size() - 1) {
                os << ", ";
            }
        }

        os << "), Scatter = (";
        for (int i = 0; i < individual.scatter().size(); ++i) {
            os << individual.scatter().at(i);
            if (i < individual.scatter().size() - 1) {
                os << ", ";
            }
        }

        os << "), Errors = (";
        for (int i = 0; i < individual.errorVector().size(); ++i) {
            os << individual.errorVector().at(i);
            if (i < individual.errorVector().size() - 1) {
                os << ", ";
            }
        }

        os << ")";
    }


    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::REvolutionaryTrainingAlgorithm &algorithm)
    {
        os
                << "REvolutionaryTrainingAlgorithm("
                << "maxNoSuccessEpochs = " << algorithm.maxNoSuccessEpochs()
                << ", populationSize = " << algorithm.populationSize()
                << ", eliteSize = " << algorithm.eliteSize()
                << ", eamin = " << algorithm.eamin()
                << ", ebmin = " << algorithm.ebmin()
                << ", ebmax = " << algorithm.ebmax();
        os << ")";
        return os;
    }

    ostream &operator<<(
            ostream &os,
            const QList<Winzent::ANN::Individual *> population)
    {
        os << "Population(";

        foreach (const Winzent::ANN::Individual *i, population) {
            os << *i << ", ";
        }

        return os << ")";
    }
}

