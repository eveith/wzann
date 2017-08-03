#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <boost/range.hpp>

#include <wzalgorithm/REvol.h>
#include <wzalgorithm/config.h>

#include "Layer.h"
#include "Connection.h"
#include "NeuralNetwork.h"

#include "TrainingSet.h"
#include "TrainingAlgorithm.h"

#include "REvolutionaryTrainingAlgorithm.h"


using std::exp;
using std::fabs;
using std::numeric_limits;

using boost::make_iterator_range;

using wzalgorithm::REvol;


namespace wzann {
    void REvolutionaryTrainingAlgorithm::getWeights(
            NeuralNetwork const& ann,
            wzalgorithm::vector_t& parameters)
    {
        for (auto const& c : make_iterator_range(ann.connections())) {
            if (c->fixedWeight()) {
                continue;
            }

            parameters.push_back(c->weight());
        }
    }


    void REvolutionaryTrainingAlgorithm::applyParameters(
            wzalgorithm::vector_t const& parameters,
            NeuralNetwork& ann)
    {
        auto pit = parameters.begin();
        auto connectionsRange = ann.connections();
        auto cit = connectionsRange.first;

        while (pit != parameters.end() && cit != connectionsRange.second) {
            if ((*cit)->fixedWeight()) {
                cit++;
                continue;
            }

            (*cit)->weight(*pit);
            cit++;
            pit++;
        }

        assert(pit == parameters.end());
        assert(cit == connectionsRange.second);
    }


    REvolutionaryTrainingAlgorithm::REvolutionaryTrainingAlgorithm():
            TrainingAlgorithm(),
            REvol()
    {
    }


    bool REvolutionaryTrainingAlgorithm::individualSucceeds(
            REvol::Individual& individual,
            NeuralNetwork& ann,
            TrainingSet const& trainingSet)
    {
        applyParameters(individual.parameters, ann);

        double error = 0.0;
        size_t numRelevantItems = 0;

        for (auto const& ti : trainingSet.trainingItems) {

            // First step: Feed forward and compare the network's
            // output with the ideal teaching output:

            const auto actual = ann.calculate(ti.input());
            if (! ti.outputRelevant()) {
                continue;
            }
            numRelevantItems++;
            auto const& expected = ti.expectedOutput();

            double lerror = 0.0;
            for (auto ait = actual.begin(), eit = expected.begin();
                    ait != actual.end() && eit != expected.end();
                    ait++, eit++) {
                lerror += std::pow(*eit - *ait, 2);
            }

            error += lerror / 2.0;
        }

        error /= static_cast<double>(numRelevantItems);
        individual.restrictions[0] = error;
        return error <= trainingSet.targetError();
    }


    void REvolutionaryTrainingAlgorithm::train(
            NeuralNetwork& ann,
            TrainingSet& trainingSet)
    {
        maxEpochs(trainingSet.maxEpochs());

        wzalgorithm::REvol::Individual origin;
        getWeights(ann, origin.parameters);
        origin.scatter.reserve(origin.parameters.size());
        for (wzalgorithm::vector_t::size_type i = 0;
                i != origin.parameters.size(); ++i) {
            origin.scatter.push_back(0.2);
        }

        auto result = REvol::run(
                origin,
                [this, &trainingSet, &ann](
                    wzalgorithm::REvol::Individual &individual) {
            return individualSucceeds(individual, ann, trainingSet);
        });

        applyParameters(result.bestIndividual.parameters, ann);
        setFinalNumEpochs(
                trainingSet,
                result.iterationsUsed);
        setFinalError(
                trainingSet,
                result.bestIndividual.restrictions.at(0));
    }
} // namespace wzann


namespace std {
    ostream &operator<<(
            ostream &os,
            wzann::REvolutionaryTrainingAlgorithm const& algorithm)
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

