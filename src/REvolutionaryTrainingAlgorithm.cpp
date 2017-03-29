#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <QPair>
#include <QList>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include <boost/random.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "REvol.h"

#include "Layer.h"
#include "Connection.h"
#include "NeuralNetwork.h"

#include "TrainingSet.h"
#include "TrainingAlgorithm.h"

#include "REvolutionaryTrainingAlgorithm.h"


using std::exp;
using std::fabs;
using std::numeric_limits;

using Winzent::Algorithm::REvol;


namespace Winzent {
    namespace ANN {
        Individual::Individual(): Algorithm::detail::Individual()
        {
        }


        Individual::Individual(
                const Algorithm::detail::Individual &individual):
                    Algorithm::detail::Individual(individual)
        {
        }


        Individual::Individual(const NeuralNetwork &neuralNetwork):
                Individual()
        {
            restrictions.push_back(numeric_limits<qreal>::infinity());

            this->parameters = parametersFromNeuralNetwork(neuralNetwork);
            this->scatter.reserve(parameters.size());

            for (auto i = 0; i != parameters.size(); ++i) {
                this->scatter.push_back(0.2);
            }
        }


        Vector Individual::parametersFromNeuralNetwork(
                const NeuralNetwork &neuralNetwork)
        {
            Vector r;

            const_cast<NeuralNetwork &>(neuralNetwork).eachConnection(
                    [&r](const Connection *const &c) {
                if (! c->fixedWeight()) {
                    r.push_back(c->weight());
                }
            });

            return r;
        }


        void Individual::applyParameters(
                const Individual &individual,
                NeuralNetwork &neuralNetwork)
        {
            int i = 0;
            neuralNetwork.eachConnection([&individual, &i](
                    Connection *const &c) {
                if (! c->fixedWeight()) {
                    c->weight(individual.parameters.at(i++));
                }
            });
        }


        Vector &Individual::errorVector()
        {
            return restrictions;
        }


        REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm():
                TrainingAlgorithm(),
                REvol()
        {
        }


        Algorithm::REvol::Population
        REvolutionaryTrainingAlgorithm::generateInitialPopulation(
                const NeuralNetwork &baseNetwork)
        {
            Individual baseIndividual(baseNetwork);
            return REvol::generateInitialPopulation(baseIndividual);
        }


        void REvolutionaryTrainingAlgorithm::evaluateIndividual(
                Algorithm::detail::Individual &individual,
                NeuralNetwork &ann,
                const TrainingSet &trainingSet)
        {
            size_t errorPos = 0;
            qreal totalMSE  = 0.0;
            individual.restrictions.resize(1);
            Individual::applyParameters(individual, ann);


            for (const auto &item: trainingSet.trainingItems) {
                const auto output = ann.calculate(item.input());
                LOG4CXX_DEBUG(
                        TrainingAlgorithm::logger,
                        "Calculated " << item.input() << " => " << output);

                if (! item.outputRelevant()) {
                    LOG4CXX_DEBUG(
                            TrainingAlgorithm::logger,
                            "Output is not relevant.");
                    continue;
                }

                qreal sampleMSE = calculateMeanSquaredError(
                        output,
                        item.expectedOutput());

                totalMSE += sampleMSE;
                errorPos += 1;
            }

            individual.restrictions[0] =
                    totalMSE / static_cast<qreal>(errorPos);
        }


        void REvolutionaryTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            maxEpochs(trainingSet.maxEpochs());

            Individual origin(ann);
            auto result = REvol::run(
                    origin,
                    [this, &trainingSet, &ann](
                        Algorithm::detail::Individual &individual) {
                evaluateIndividual(individual, ann, trainingSet);
                return individual.restrictions.at(0)
                    <= trainingSet.targetError();
            });

            Individual::applyParameters(result.bestIndividual, ann);
            setFinalNumEpochs(
                    trainingSet,
                    result.iterationsUsed);
            setFinalError(
                    trainingSet,
                    result.bestIndividual.restrictions.at(0));
        }
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::REvolutionaryTrainingAlgorithm &algorithm)
    {
        os
                << "REvolutionaryTrainingAlgorithm("
                << "maxNoSuccessEpochs = " << algorithm.maxNoSuccessEpochs()
                << ", populationSize = " << algorithm.populationSize()
                << ", eliteSize = " << algorithm.eliteSize()
                << ", startTTL = " << algorithm.startTTL()
                << ", gradientWeight = " << algorithm.gradientWeight()
                << ", successWeight = " << algorithm.successWeight()
                << ", eamin = " << algorithm.eamin()
                << ", ebmin = " << algorithm.ebmin()
                << ", ebmax = " << algorithm.ebmax();
        os << ")";
        return os;
    }
}
