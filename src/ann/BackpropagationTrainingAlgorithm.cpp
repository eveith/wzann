/*!
 * \file
 */


#include <limits>
#include <cmath>

#include <QHash>
#include <QDebug>

#include "Neuron.h"
#include "Layer.h"
#include "Connection.h"
#include "ActivationFunction.h"
#include "NeuralNetwork.h"
#include "TrainingSet.h"
#include "Exception.h"

#include "BackpropagationTrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {

        BackpropagationTrainingAlgorithm::BackpropagationTrainingAlgorithm(
                qreal learningRate,
                QObject* parent):
                    TrainingAlgorithm(parent),
                    m_learningRate(learningRate)
        {
        }


        qreal BackpropagationTrainingAlgorithm::learningRate() const
        {
            return m_learningRate;
        }


        BackpropagationTrainingAlgorithm &
        BackpropagationTrainingAlgorithm::learningRate(const qreal &rate)
        {
            m_learningRate = rate;
            return *this;
        }


        void BackpropagationTrainingAlgorithm::train(
                NeuralNetwork *const &ann,
                TrainingSet &trainingSet)
        {
            // Initialize the state variables:

            int epochs  = 0;
            qreal error = std::numeric_limits<qreal>::max();

            for(; epochs < trainingSet.maxEpochs()
                        && error > trainingSet.targetError();
                    ++epochs) {

                // Begin this run with an error of 0, and clear the state
                // variables:

                error = 0.0;

                // We present each training pattern once per epoch. An epoch
                // is complete once we've presented the complete training set to
                // the network. We then calculate the overall error and try to
                // minimize it.

                QList<TrainingItem>::const_iterator it =
                        trainingSet.trainingData().constBegin();
                for (; it != trainingSet.trainingData().constEnd(); it++) {

                    // Reset memoization fields:

                    m_deltas.clear();
                    m_outputError.clear();

                    // Set up state storage:

                    QHash<Neuron*, qreal> neuronDeltas;
                    QHash<Connection*, qreal> connectionDeltas;

                    // First step: Feed forward and compare the network's output
                    // with the ideal teaching output:

                    ValueVector actualOutput =
                            ann->calculate(it->input());
                    ValueVector expectedOutput = it->expectedOutput();
                    ValueVector errorOutput = outputError(
                            actualOutput,
                            expectedOutput);

                    // Add squared error for this run:

                    for (const auto &d: errorOutput) {
                        error += d*d;
                    }

                    // Backpropagation works in two phases: First, the error is
                    // propagated backwarts from the output layer and the
                    // weight deltas are calculated, then the deltas are
                    // applied.

                    ann->eachConnection([&](Connection *const &c) {
                        auto dn = neuronDelta(
                                ann,
                                c->destination(),
                                neuronDeltas,
                                errorOutput);
                        connectionDeltas.insert(
                                c,
                                learningRate() * dn
                                    * c->source()->lastResult());
                    });

                    foreach (Connection *c, connectionDeltas.keys()) {
                        c->weight(c->weight() + connectionDeltas[c]);
                    }
                }

                // It's called MEAN square error for a reason:

                error /= (trainingSet.trainingData().count()
                        * trainingSet.trainingData().first()
                            .expectedOutput().count());
            }

            // Store final training results:

            setFinalError(trainingSet, error);
            setFinalNumEpochs(trainingSet, epochs);
        }


        ValueVector BackpropagationTrainingAlgorithm::outputError(
                const ValueVector &actual,
                const ValueVector &expected)
                    const
        {
            if (actual.size() != expected.size()) {
                throw LayerSizeMismatchException(
                        actual.size(),
                        expected.size());
            }

            ValueVector error;

            for (int i = 0; i != actual.size(); ++i) {
                error << expected[i] - actual[i];
            }

            return error;
        }


        qreal BackpropagationTrainingAlgorithm::outputNeuronDelta(
                const Neuron *neuron,
                const qreal &error)
                    const
        {
            return neuron->activationFunction()->calculateDerivative(
                            neuron->lastInput(),
                            neuron->lastResult())
                    * error;
        }


        qreal BackpropagationTrainingAlgorithm::hiddenNeuronDelta(
                NeuralNetwork *const &ann,
                Neuron *neuron,
                QHash<Neuron *, qreal> &neuronDeltas,
                const ValueVector &outputError)
                    const
        {
            qreal delta = 0.0;

            const auto connections = ann->neuronConnectionsFrom(neuron);
            Q_ASSERT(connections.size() > 0);

            for (const auto &c: connections) {
                // weight(j,k) * delta(k):
                delta += neuronDelta(
                            ann,
                            c->destination(),
                            neuronDeltas,
                            outputError)
                        * c->weight();
                delta += neuronDeltas[c->destination()] * c->weight();
            }

            delta *= neuron->activationFunction()->calculateDerivative(
                    neuron->lastInput(),
                    neuron->lastResult());

            return delta;
        }


        qreal BackpropagationTrainingAlgorithm::neuronDelta(
                NeuralNetwork *const &ann,
                Neuron *neuron,
                QHash<Neuron *, qreal> &neuronDeltas,
                const ValueVector &outputError)
                    const
        {
            Q_ASSERT(! ann->inputLayer()->contains(neuron));

            if (neuronDeltas.contains(neuron)) {
                return neuronDeltas[neuron];
            }

            if (ann->outputLayer()->contains(neuron)) {
                int neuronIdx = ann->outputLayer()->indexOf(neuron);
                neuronDeltas.insert(
                        neuron,
                        outputNeuronDelta(neuron, outputError.at(neuronIdx)));
            } else {
                neuronDeltas.insert(
                        neuron,
                        hiddenNeuronDelta(
                            ann,
                            neuron,
                            neuronDeltas,
                            outputError));
            }

            return neuronDeltas[neuron];
        }
    }
}
