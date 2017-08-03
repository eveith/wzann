#include <limits>
#include <cmath>
#include <cstddef>
#include <algorithm>

#include <log4cxx/logger.h>

#include <QtDebug>
#include <QTextStream>

#include "QtContainerOutput.h"

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


        const double SimulatedAnnealingTrainingAlgorithm::CUT = 0.5;


        SimulatedAnnealingTrainingAlgorithm::SimulatedAnnealingTrainingAlgorithm(
                double startTemperature,
                double stopTemperature,
                size_t cycles):
                    TrainingAlgorithm(),
                    m_startTemperature(startTemperature),
                    m_stopTemperature(stopTemperature),
                    m_cycles(cycles)
        {
        }


        double SimulatedAnnealingTrainingAlgorithm::startTemperature() const
        {
            return m_startTemperature;
        }


        double SimulatedAnnealingTrainingAlgorithm::stopTemperature() const
        {
            return m_stopTemperature;
        }


        size_t SimulatedAnnealingTrainingAlgorithm::cycles() const
        {
            return m_cycles;
        }


        Vector SimulatedAnnealingTrainingAlgorithm::getParameters(
                const NeuralNetwork &neuralNetwork)
        {
            Vector r;

            const_cast<NeuralNetwork &>(neuralNetwork).eachConnection(
                        [&r](const Connection *const &c) {
                if (!c->fixedWeight()) {
                    r << c->weight();
                }
            });

            return r;
        }


        void SimulatedAnnealingTrainingAlgorithm::applyParameters(
                const Vector &parameters,
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
                Vector &parameters,
                const double &temperature)
        {

            std::for_each(parameters.begin(), parameters.end(),
                        [this, &temperature](double &w) {
                double add = CUT - qrand() / static_cast<double>(RAND_MAX);
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


        double SimulatedAnnealingTrainingAlgorithm::iterate(
                NeuralNetwork &network,
                TrainingSet const &trainingSet)
        {
            // Initialze state: Safe the best known network configuration and
            // the score (i. e., error value) of that network:

            Vector bestParameters;
            double bestScore     = std::numeric_limits<double>::max();
            double temperature   = startTemperature();

            // Execute all circles, plus one to get the score of the current
            // solution:

            for (auto i = 0; i < cycles(); ++i) {
                double score         = 0.0;
                size_t trainingItems= 0;

                Vector parameters = getParameters(network);
                randomize(parameters, temperature);
                applyParameters(parameters, network);

                for (const auto &item: trainingSet.trainingItems) {
                    Vector actualOutput = network.calculate(
                            item.input());

                    if (! item.outputRelevant()) {
                        continue;
                    }

                    Vector expectedOutput = item.expectedOutput();
                    score += calculateMeanSquaredError(
                            actualOutput,
                            expectedOutput);
                    ++trainingItems;
                }

                score /= static_cast<double>(trainingItems);

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
                        / static_cast<double>(cycles() - 1));
            }

            applyParameters(bestParameters, network);
            return bestScore;
        }


        void SimulatedAnnealingTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            // Init state:

            double error     = std::numeric_limits<double>::max();
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
