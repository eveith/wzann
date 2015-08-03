#include <limits>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <QPair>
#include <QList>

#include <log4cxx/logger.h>

#include <boost/random.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "Layer.h"
#include "Connection.h"
#include "NeuralNetwork.h"

#include "TrainingSet.h"
#include "TrainingAlgorithm.h"

#include "REvolutionaryTrainingAlgorithm.h"


#pragma STDC FENV_ACCESS ON


using std::exp;
using std::fabs;
using std::numeric_limits;


namespace Winzent {
    namespace ANN {
        Individual::Individual(): m_timeToLive(0)
        {
        }


        Individual::Individual(const NeuralNetwork &neuralNetwork):
                Individual()
        {
            m_errorVector.push_back(numeric_limits<qreal>::infinity());

            m_parameters = parameters(neuralNetwork);
            m_scatter.reserve(m_parameters.size());

            for (auto i = 0; i != m_parameters.size(); ++i) {
                m_scatter.push_back(0.2);
            }
        }


        Individual::Individual(const Vector &parameters): Individual()
        {
            m_parameters = parameters;
            m_scatter.fill(0.2, m_parameters.size());
        }


        Vector Individual::scatter() const
        {
            return m_scatter;
        }


        Vector &Individual::scatter()
        {
            return m_scatter;
        }


        Individual &Individual::scatter(Vector scatter)
        {
            m_scatter = scatter;
            return *this;
        }


        const Vector &Individual::parameters() const
        {
            return m_parameters;
        }


        Vector &Individual::parameters()
        {
            return m_parameters;
        }


        Individual &Individual::parameters(const Vector &parameters)
        {
            m_parameters = parameters;
            return *this;
        }


        Vector Individual::parameters(const NeuralNetwork &neuralNetwork)
                const
        {
            Vector r;

            neuralNetwork.eachConnection([&r](const Connection *const &c) {
                if (!c->fixedWeight()) {
                    r.push_back(c->weight());
                }
            });

            return r;
        }


        void Individual::applyParameters(NeuralNetwork &neuralNetwork) const
        {
            int i = 0;
            neuralNetwork.eachConnection([this, &i](Connection *const &c) {
                if (!c->fixedWeight()) {
                    c->weight(m_parameters.at(i++));
                }
            });
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


        Vector &Individual::errorVector()
        {
            return m_errorVector;
        }


        const Vector &Individual::errorVector() const
        {
            return m_errorVector;
        }


        int Individual::compare(const Individual &other) const
        {
            if (this->timeToLive() < 0 && other.timeToLive() >= 0) {
                return -1;
            }

            if (this->timeToLive() >= 0 && other.timeToLive() < 0) {
                return 1;
            }

            if (this->errorVector().at(0) < other.errorVector().at(0)) {
                return 1;
            } else if (errorVector().at(0) > other.errorVector().at(0)) {
                return -1;
            } else {
                auto size = (
                        errorVector().size() > other.errorVector().size()
                        ? other.errorVector().size()
                        : this->errorVector().size());
                for (auto i = 1; i < size; ++i) {
                    if (errorVector().at(i) < other.errorVector().at(i)) {
                        return 1;
                    } else if (other.errorVector().at(i)
                            < this->errorVector().at(i)) {
                        return -1;
                    }
                }
            }

            return 0;
        }


        bool Individual::isBetterThan(const Individual &other) const
        {
            return (1 == compare(other));
        }


        bool Individual::isIndividual1Better(
                Individual const& i1,
                Individual const& i2)
        {
            return i1.isBetterThan(i2);
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
            m_parameters    = other.m_parameters;
            m_scatter       = other.m_scatter;
            m_errorVector   = other.m_errorVector;

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
                Population& population)
        {
            population.sort(&Individual::isIndividual1Better);
        }


        qreal REvolutionaryTrainingAlgorithm::frandom()
        {
            return m_uniformDistribution(m_randomNumberGenerator);
        }


        REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm():
                TrainingAlgorithm(),
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


        size_t REvolutionaryTrainingAlgorithm::maxNoSuccessEpochs() const
        {
            return m_maxNoSuccessEpochs;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::maxNoSuccessEpochs(
                const size_t &epochs)
        {
            m_maxNoSuccessEpochs = epochs;
            return *this;
        }


        size_t REvolutionaryTrainingAlgorithm::populationSize() const
        {
            return m_populationSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::populationSize(const size_t &size)
        {
            m_populationSize = size;

            if (0 == eliteSize()) {
                eliteSize(std::ceil(size * 0.1));
            }

            return *this;
        }


        size_t REvolutionaryTrainingAlgorithm::eliteSize() const
        {
            return m_eliteSize;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::eliteSize(const size_t &size)
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


        size_t REvolutionaryTrainingAlgorithm::measurementEpochs() const
        {
            return m_measurementEpochs;
        }


        REvolutionaryTrainingAlgorithm &
        REvolutionaryTrainingAlgorithm::measurementEpochs(
                const size_t &epochs)
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
                LOG4CXX_ERROR(
                        logger,
                        "Elite is bigger or equal to population");
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


        REvolutionaryTrainingAlgorithm::Population
        REvolutionaryTrainingAlgorithm::generateInitialPopulation(
                const NeuralNetwork &baseNetwork)
        {
            Individual *baseIndividual = new Individual(baseNetwork);
            baseIndividual->timeToLive(startTTL());

            Population population;
            population.push_back(baseIndividual);
            auto numParameters = baseIndividual->parameters().size();

            for (auto i = 1; i < populationSize() + 1; ++i) {
                Individual *individual = new Individual();
                individual->timeToLive(startTTL());
                individual->parameters().reserve(numParameters);
                individual->scatter().reserve(numParameters);

                for (auto j = 0; j != numParameters; ++j) {
                    qreal r = baseIndividual->scatter().at(j) * exp(
                            0.4 * (0.5 - frandom()));
                    individual->scatter().push_back(r);
                    individual->parameters().push_back(
                            baseIndividual->parameters().at(j)
                                + r * (frandom() - frandom()
                                    + frandom() - frandom()));
                }

                population.push_back(individual);
            }

            Q_ASSERT(population.size() == populationSize() + 1);
            return population;
        }


        QPair<Individual &, Individual &>
        REvolutionaryTrainingAlgorithm::modifyIndividual(
                Individual& individual,
                Population& population)
        {
            boost::uniform_int<> rnDistribution;

            Individual &eliteIndividual = population.at(abs(
                    (rnDistribution(m_randomNumberGenerator) % eliteSize())
                        - (rnDistribution(m_randomNumberGenerator)
                            % eliteSize())));
            Individual &otherIndividual = population.at(
                    rnDistribution(m_randomNumberGenerator)
                        % population.size());

            if (otherIndividual.isBetterThan(eliteIndividual)) {
                Individual &tmp = eliteIndividual;
                eliteIndividual = otherIndividual;
                otherIndividual = tmp;
            }

            qreal xlp = 0.0;
            qreal successRate = m_success / m_targetSuccess - 1.0;
            int gradientSwitch = rnDistribution(m_randomNumberGenerator) % 3;
            qreal expvar = exp(frandom() - frandom());

            if (2 == gradientSwitch) {
                xlp = (frandom() + frandom() + frandom() + frandom()
                        + frandom() + frandom() + frandom() + frandom()
                        + frandom() + frandom() - frandom() - frandom()
                        - frandom() - frandom() - frandom() - frandom())
                    * gradientWeight();

                if (xlp > 0.0) {
                    xlp *= 0.5;
                }

                xlp *= exp(successWeight() * successRate);
            }

            // Now modify the new individual:

            auto numParameters = eliteIndividual.parameters().size();
            Vector newParameters;
            newParameters.reserve(numParameters);

            Q_ASSERT(numParameters == eliteIndividual.parameters().size());
            Q_ASSERT(numParameters == otherIndividual.parameters().size());

            for (auto i = 0; i != numParameters; ++i) {
                std::feclearexcept(FE_ALL_EXCEPT);

                qreal dx = eliteIndividual.scatter().at(i) * exp(
                        successWeight() * successRate);

                dx = applyDxBounds(dx, eliteIndividual.parameters().at(i));

                // Mutate scatter:

                eliteIndividual.scatter()[i] = dx;

                if (frandom() < 0.5) {
                    dx = eliteIndividual.scatter().at(i);
                } else {
                    dx = 0.5 * (eliteIndividual.scatter().at(i)
                            + otherIndividual.scatter().at(i));
                }

                dx *= expvar;
                dx = applyDxBounds(dx, eliteIndividual.parameters().at(i));

                // Generate new scatter:

                individual.scatter()[i] = dx;

                dx = dx * (frandom() + frandom()
                        + frandom() + frandom() + frandom() - frandom()
                        - frandom() - frandom() - frandom() - frandom());

                if (0 == gradientSwitch) { // Everything from the elite, p=2/3
                    if (rnDistribution(m_randomNumberGenerator) % 3 < 2) {
                        dx += eliteIndividual.parameters().at(i);
                    } else {
                        dx += otherIndividual.parameters().at(i);
                    }
                } else if (1 == gradientSwitch) { // use eliteIndividual
                    dx += eliteIndividual.parameters().at(i);
                } else if (2 == gradientSwitch) { // use elite & gradient
                    dx += eliteIndividual.parameters().at(i);
                    dx += xlp * (eliteIndividual.parameters().at(i)
                            - otherIndividual.parameters().at(i));
                }

                newParameters.push_back(dx);
            }

            Q_ASSERT(newParameters.size() == individual.parameters().size());
            individual.parameters(newParameters);
            individual.timeToLive(startTTL());
            individual.errorVector()[0] = numeric_limits<qreal>::infinity();

#ifdef      QT_DEBUG
                for (auto i = 0; i != newParameters.size(); ++i) {
                    Q_ASSERT(newParameters.at(i)
                             == individual.parameters().at(i));
                }
#endif

            LOG4CXX_DEBUG(logger, "Created " << individual);

            return QPair<Individual &, Individual &>(
                    eliteIndividual,
                    otherIndividual);
        }


        void REvolutionaryTrainingAlgorithm::evaluateIndividual(
                Individual &individual,
                NeuralNetwork &ann,
                const TrainingSet &trainingSet)
        {
            size_t errorPos = 1;
            qreal totalMSE  = 0.0;


            for (const auto &item: trainingSet.trainingData()) {
                individual.applyParameters(ann);
                Vector output = ann.calculate(item.input());

                if (! item.outputRelevant()) {
                    continue;
                }

                qreal sampleMSE = calculateMeanSquaredError(
                        output,
                        item.expectedOutput());

                individual.errorVector()[errorPos++] = sampleMSE;
                totalMSE += sampleMSE;
            }

            individual.errorVector()[0] =
                    totalMSE / static_cast<qreal>(errorPos);
            individual.age();
        }


        void REvolutionaryTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            if (0 == startTTL()) {
                startTTL(std::ceil(trainingSet.maxEpochs() * 0.1));
            }

            if (0 == measurementEpochs()) {
                measurementEpochs(std::ceil(trainingSet.maxEpochs() / 200.0));
            }

            if (!hasSensibleTrainingParameters()) {
                LOG4CXX_ERROR(
                        logger,
                        "Training parameters have no sensible values, "
                            "won't train.");
                return;
            }

            size_t lastSuccess = 0;
            size_t epoch       = 0;
            Population population = generateInitialPopulation(ann);
            Individual& bestIndividual = population.front();

            do {
                // Modify the worst individual:

                if (0 != epoch) {
                    auto individual = population.pop_back();
                    auto srcIndividuals = modifyIndividual(
                            *individual,
                            population);

                    evaluateIndividual(
                            *individual,
                            ann,
                            trainingSet);
                    evaluateIndividual(
                            srcIndividuals.first,
                            ann,
                            trainingSet);
                    evaluateIndividual(
                            srcIndividuals.second,
                            ann,
                            trainingSet);

                    population.push_back(individual.release());
                } else {
                    for (auto &individual: population) {
                        individual.errorVector().fill(
                                0.0,
                                1 + trainingSet.trainingData().size());
                        evaluateIndividual(individual, ann, trainingSet);
                    }

                    sortPopulation(population);
                }

                // Check for addition of a new individual:

                Individual &newIndividual = population.back();
                Individual &worstIndividual = population.at(
                        population.size() - 2);

                // Check for global or, at least, local improvement:

                if (newIndividual.isBetterThan(bestIndividual)) {
                    lastSuccess = epoch;
                    bestIndividual = newIndividual;
                    bestIndividual.timeToLive(epoch);
                } else {
                    if (newIndividual.isBetterThan(worstIndividual)) {
                        if (worstIndividual.timeToLive() >= 0) {
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
                            << ", bestIndividual = " << bestIndividual
                            << ")");
            } while (population.front().errorVector().front()
                        > trainingSet.targetError()
                    && epoch < trainingSet.maxEpochs()
                    && epoch - lastSuccess < maxNoSuccessEpochs());

            bestIndividual.applyParameters(ann);
            setFinalNumEpochs(trainingSet, epoch);
            setFinalError(trainingSet, bestIndividual.errorVector().at(0));

            LOG4CXX_DEBUG(
                    logger,
                    "Training ended after " << epoch << " epochs; "
                        << population.front());
        }
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::Individual &individual)
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

        return os;
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
}

