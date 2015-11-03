#include <limits>
#include <cstddef>
#include <algorithm>
#include <functional>

#include "Connection.h"
#include "NeuralNetwork.h"

#include "TrainingAlgorithm.h"
#include "ParticleSwarmOptimization.h"

#include "PsoTrainingAlgorithm.h"

#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        PsoTrainingAlgorithm::PsoTrainingAlgorithm():
                TrainingAlgorithm(),
                Algorithm::ParticleSwarmOptimization()
        {
            lowerBoundary(-std::numeric_limits<qreal>::max());
            upperBoundary(std::numeric_limits<qreal>::max());
        }


        void PsoTrainingAlgorithm::applyPosition(
                const Algorithm::detail::Particle &particle,
                NeuralNetwork &neuralNetwork)
        {
            size_t i = 0;
            neuralNetwork.eachConnection([&i, &particle](
                    Connection *const &connection) {
                if (connection->fixedWeight()) {
                    return;
                }

                connection->weight(particle.currentPosition.at(i++));
            });
        }


        void PsoTrainingAlgorithm::evaluateParticle(
                Algorithm::detail::Particle &particle,
                NeuralNetwork &ann,
                const TrainingSet &trainingSet)
        {
            applyPosition(particle, ann);
            qreal errorSum = 0;
            size_t i = 0;

            for (const auto &item: trainingSet.trainingData()) {
                auto actual = ann.calculate(item.input());

                if (! item.outputRelevant()) {
                    continue;
                }

                auto expected = item.expectedOutput();
                errorSum += calculateMeanSquaredError(actual, expected);
                i += 1;
            }

            particle.currentFitness = errorSum / static_cast<qreal>(i);
        }


        void PsoTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            size_t dimensions = 0;
            ann.eachConnection([&dimensions](const Connection *const &c) {
                if (! c->fixedWeight()) {
                    ++dimensions;
                }
            });

            auto result = run(
                    dimensions,
                    [this, &ann, &trainingSet](
                        Algorithm::detail::Particle &particle) {
                evaluateParticle(particle, ann, trainingSet);
                return particle.currentFitness <= trainingSet.targetError();
            });

            setFinalError(trainingSet, result.bestParticle.bestFitness);
            setFinalNumEpochs(trainingSet, result.iterationsUsed);
            applyPosition(result.bestParticle, ann);
        }
    }
}
