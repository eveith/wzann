#include <limits>
#include <cmath>
#include <cstddef>
#include <algorithm>

#include <log4cxx/logger.h>

#include <QtDebug>
#include <QTextStream>

#include "TrainingAlgorithm.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "TrainingSet.h"
#include "SimulatedAnnealingTrainingAlgorithm.h"


using std::exp;
using std::log;


namespace Winzent {
    namespace ANN {


        const qreal SimulatedAnnealingTrainingAlgorithm::CUT = 0.5;


        SimulatedAnnealingTrainingAlgorithm::SimulatedAnnealingTrainingAlgorithm(
                qreal startTemperature,
                qreal stopTemperature,
                size_t cycles):
                    TrainingAlgorithm(),
                    m_startTemperature(startTemperature),
                    m_stopTemperature(stopTemperature),
                    m_cycles(cycles)
        {
        }


        qreal SimulatedAnnealingTrainingAlgorithm::startTemperature() const
        {
            return m_startTemperature;
        }


        qreal SimulatedAnnealingTrainingAlgorithm::stopTemperature() const
        {
            return m_stopTemperature;
        }


        size_t SimulatedAnnealingTrainingAlgorithm::cycles() const
        {
            return m_cycles;
        }


        ValueVector SimulatedAnnealingTrainingAlgorithm::getParameters(
                const NeuralNetwork &neuralNetwork)
        {
            ValueVector r;

            neuralNetwork.eachConnection([&r](const Connection *const &c) {
                if (!c->fixedWeight()) {
                    r << c->weight();
                }
            });

            return r;
        }


        void SimulatedAnnealingTrainingAlgorithm::applyParameters(
                const ValueVector &parameters,
                NeuralNetwork &neuralNetwork)
        {
            int i = 0;
            neuralNetwork.eachConnection([&i, &parameters](
                    Connection *const &c) {
                if (!c->fixedWeight()) {
                    c->weight(parameters.at(i++));
                }
            });
        }


        void SimulatedAnnealingTrainingAlgorithm::randomize(
                ValueVector &parameters,
                const qreal &temperature)
        {

            std::for_each(parameters.begin(), parameters.end(),
                        [this, &temperature](qreal &w) {
                qreal add = CUT - qrand() / static_cast<qreal>(RAND_MAX);
                add /= startTemperature();
                add *= temperature;

                w += add;

                LOG4CXX_DEBUG(
                        logger,
                         "w = "
                            << w - add
                            << " + "
                            << add
                            << " = "
                            << w);
            });
        }


        qreal SimulatedAnnealingTrainingAlgorithm::iterate(
                NeuralNetwork &network,
                TrainingSet const &trainingSet)
        {
            // Initialze state: Safe the best known network configuration and
            // the score (i. e., error value) of that network:

            ValueVector bestParameters;
            qreal bestScore     = std::numeric_limits<qreal>::max();
            qreal temperature   = startTemperature();

            // Execute all circles, plus one to get the score of the current
            // solution:

            for (auto i = 0; i < cycles(); ++i) {
                qreal score         = 0.0;
                size_t trainingItems= 0;

                ValueVector parameters = getParameters(network);
                randomize(parameters, temperature);
                applyParameters(parameters, network);

                for (const auto &item: trainingSet.trainingData()) {
                    ValueVector actualOutput = network.calculate(
                            item.input());

                    if (! item.outputRelevant()) {
                        continue;
                    }

                    ValueVector expectedOutput = item.expectedOutput();
                    score += calculateMeanSquaredError(
                            actualOutput,
                            expectedOutput);
                    ++trainingItems;
                }

                score /= static_cast<qreal>(trainingItems);

                // Accept the solution if its better (score < bestScore)

                LOG4CXX_DEBUG(
                        logger,
                        "Score: "
                            << score
                            << ", bestScore: "
                            << bestScore);

                if (score < bestScore) {
                    bestParameters  = parameters;
                    bestScore       = score;

                    LOG4CXX_DEBUG(
                            logger,
                            "Accepted solution " << bestParameters
                                << ", score:" << score);
                }

                temperature *= exp(log(stopTemperature() / startTemperature())
                        / static_cast<qreal>(cycles() - 1));
            }

            applyParameters(bestParameters, network);
            return bestScore;
        }


        void SimulatedAnnealingTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            // Init state:

            qreal error     = std::numeric_limits<qreal>::max();
            size_t epoch    = -1;

            while (error > trainingSet.targetError()
                   && ++epoch < trainingSet.maxEpochs()) {
                error = iterate(ann, trainingSet);

                LOG4CXX_DEBUG(
                        logger,
                        "Epoch #" << epoch << ": "
                            << "error: " << error
                            << " targetError: " << trainingSet.targetError());

            }

            // We're done, restore the cache size and save information:

            setFinalNumEpochs(trainingSet, epoch);
            setFinalError(trainingSet, error);
        }
    }
}
