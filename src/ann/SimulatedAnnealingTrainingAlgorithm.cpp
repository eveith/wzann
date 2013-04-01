#include <QObject>

#include <limits>
#include <cmath>

#include <QtDebug>
#include <cstdio>
#include <QTextStream>

#include "TrainingAlgorithm.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "TrainingSet.h"
#include "SimulatedAnnealingTrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        const qreal SimulatedAnnealingTrainingAlgorithm::CUT = 0.5;


        SimulatedAnnealingTrainingAlgorithm::SimulatedAnnealingTrainingAlgorithm(
                qreal startTemperature,
                qreal stopTemperature,
                int cycles,
                QObject *parent):
                    TrainingAlgorithm(parent),
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


        int SimulatedAnnealingTrainingAlgorithm::cycles() const
        {
            return m_cycles;
        }


        void SimulatedAnnealingTrainingAlgorithm::randomize(
                NeuralNetwork *network,
                const qreal &temperature)
        {
            for (int i = 0; i != network->size(); ++i) {
                Layer *layer = network->layerAt(i);

                for (int j = 0; j <= layer->size(); ++j) {
                    Neuron *neuron = layer->neuronAt(j);
                    QList<Connection *> connections =
                            network->neuronConnectionsFrom(neuron);
                    foreach (Connection *connection, connections) {
                        if (connection->fixedWeight()) {
                            continue;
                        }

                        qreal add = CUT-qrand() / static_cast<qreal>(RAND_MAX);
                        add /= startTemperature();
                        add *= temperature;

                        qDebug() << "randomize() add:" << add;

                        connection->weight(connection->weight() + add);
                        qDebug()
                                << connection
                                << "old weight:" << connection->weight() - add
                                << "new weight:" << connection->weight();
                    }

                }
            }
        }


        qreal SimulatedAnnealingTrainingAlgorithm::iterate(
                NeuralNetwork *&network,
                TrainingSet *trainingSet)
        {
            // Initialze state: Safe the best known network configuration and
            // the score (i. e., error value) of that network:

            NeuralNetwork *best = NULL;
            qreal bestScore = std::numeric_limits<qreal>::max();
            qreal temperature = startTemperature();

            // Execute all circles, plus one to get the score of the current
            // solution:

            for (int i = 0; i <= cycles(); ++i) {
                qreal score = 0.0;

                foreach (TrainingItem item, trainingSet->trainingData()) {
                    ValueVector actualOutput = network->calculate(item.input());
                    ValueVector expectedOutput = item.expectedOutput();

                    for (int k = 0; k != expectedOutput.size(); ++k) {
                        score += std::pow(
                                expectedOutput[k] - actualOutput[k],
                                2);
                    }
                }

                score /= static_cast<qreal>(
                        trainingSet->trainingData().count()
                        * trainingSet->trainingData().first()
                            .expectedOutput().count());

                // Accept the solution if its better (score < bestScore)

                qDebug()
                        << "Score:" << score
                        << "bestScore:" << bestScore
                        << "Accept worse probability:"
                        << std::exp(- ((score-bestScore) / temperature));

                if (score < bestScore) {
                    if (NULL != best) {
                        delete best;
                    }
                    best = network->clone();
                    bestScore = score;

                    qDebug()
                            << "Accepted solution" << best
                            << "score:" << score;
                }

                randomize(network, temperature);

                qDebug()
                        << "Temperature:" << temperature
                        << "Cycles: " << i;

                temperature *= std::exp(
                        std::log(stopTemperature() / startTemperature())
                        / (cycles() - 1));
            }

            network = best;
            return bestScore;
        }


        void SimulatedAnnealingTrainingAlgorithm::train(
                NeuralNetwork *const network,
                TrainingSet *trainingSet)
        {
            // We do not need any caching here:

            setNeuronCacheSize(network, 0);


            // Init state:

            qreal error = std::numeric_limits<double>::max();
            int epoch = 0;
            NeuralNetwork *solution = network->clone();

            while (error > trainingSet->targetError()
                   && epoch < trainingSet->maxEpochs()) {
                error = iterate(solution, trainingSet);

                epoch++;
                qDebug()
                        << "error:" << error
                        << "targetError:" << trainingSet->targetError()
                        << "epoch:" << epoch;
            }

            // Copy weights:

            for (int i = 0; i != network->size(); ++i) {
                Layer *origLayer = network->layerAt(i);
                Layer *newLayer = solution->layerAt(i);

                for (int j = 0; j <= origLayer->size(); ++j) {
                    Neuron *origNeuron = origLayer->neuronAt(j);
                    Neuron *newNeuron = newLayer->neuronAt(j);

                    QList<Connection *> origConnections =
                            network->neuronConnectionsFrom(origNeuron);
                    QList<Connection *> newConnections =
                            solution->neuronConnectionsFrom(newNeuron);
                    Q_ASSERT(origConnections.size() == newConnections.size());

                    for (int k = 0; k != origConnections.size(); ++k) {
                        if (origConnections[k]->fixedWeight()) {
                            continue;
                        }

                        origConnections[k]->weight(newConnections[k]->weight());
                    }
                }
            }

            // We're done, restore the cache size:

            restoreNeuronCacheSize();
        }
    }
}
