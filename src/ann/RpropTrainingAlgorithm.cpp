#include <QHash>

#include <cmath>
#include <limits>
#include <algorithm>

#include <boost/range.hpp>

#include <log4cxx/logger.h>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "Layer.h"
#include "Connection.h"

#include "Neuron.h"
#include "ActivationFunction.h"

#include "TrainingAlgorithm.h"
#include "RpropTrainingAlgorithm.h"


using std::accumulate;
using std::fabs;
using std::max;
using std::min;

using boost::make_iterator_range;


namespace Winzent {
    namespace ANN {



        const qreal RpropTrainingAlgorithm::ETA_POSITIVE =  1.2;
        const qreal RpropTrainingAlgorithm::ETA_NEGATIVE = -0.5;
        const qreal RpropTrainingAlgorithm::ZERO_TOLERANCE =
                0.00000000000000001;
        const qreal RpropTrainingAlgorithm::DEFAULT_INITIAL_UPDATE = 0.1;
        const qreal RpropTrainingAlgorithm::DELTA_MIN = 1e-6;
        const qreal RpropTrainingAlgorithm::MAX_STEP = 50.0;



        RpropTrainingAlgorithm::RpropTrainingAlgorithm(): TrainingAlgorithm()
        {
        }


        int RpropTrainingAlgorithm::sgn(const qreal &x)
        {
            if (fabs(x) < ZERO_TOLERANCE) {
                return 0;
            }

            if (x < 0.0) {
                return -1;
            }

            return 1;
        }


        Vector RpropTrainingAlgorithm::feedForward(
                NeuralNetwork &network,
                const TrainingItem &trainingItem)
                const
        {
            Vector actualOutput = network.calculate(
                    trainingItem.input());
            return outputError(trainingItem.expectedOutput(), actualOutput);
        }


        Vector RpropTrainingAlgorithm::outputError(
                const Vector &expected,
                const Vector &actual)
                const
        {
            Q_ASSERT(expected.size() == actual.size());

            Vector error;
            error.reserve(actual.size());

            for (auto i = 0; i != actual.size(); ++i) {
                error.push_back(expected.at(i) - actual.at(i));
            }

            return error;
        }


        qreal RpropTrainingAlgorithm::outputNeuronDelta(
                const Neuron &neuron,
                const qreal &error)
                const
        {
            return error * neuron.activationFunction()->calculateDerivative(
                    neuron.lastInput(),
                    neuron.lastResult());
        }


        qreal RpropTrainingAlgorithm::hiddenNeuronDelta(
                NeuralNetwork &ann,
                const Neuron &neuron,
                QHash<const Neuron *, qreal> &neuronDeltas,
                const Vector &outputError)
                const
        {
            qreal delta = 0.0;

            auto connections = ann.connectionsFrom(neuron);
            Q_ASSERT(connections.second-connections.first > 0);

            for (const auto &c: make_iterator_range(connections)) {
                // weight(j,k) * delta(k):
                delta += neuronDelta(
                            ann,
                            c->destination(),
                            neuronDeltas,
                            outputError)
                        * c->weight();
                delta += neuronDeltas.value(&(c->destination())) *c->weight();
            }

            delta *= neuron.activationFunction()->calculateDerivative(
                    neuron.lastInput(),
                    neuron.lastResult());

            return delta;
        }


        qreal RpropTrainingAlgorithm::neuronDelta(
                NeuralNetwork &ann,
                const Neuron &neuron,
                QHash<const Neuron *, qreal> &neuronDeltas,
                const Vector &outputError)
                const
        {
            // Apply memoization, where possible:

            if (neuronDeltas.contains(&neuron)) {
                return neuronDeltas.value(&neuron);
            }

            // What layer does the neuron live in?

            Q_ASSERT(! ann.inputLayer().contains(neuron));
            qreal delta = 0.0;

            if (ann.outputLayer().contains(neuron)) {
                qreal error = outputError.at(
                        ann.outputLayer().indexOf(neuron));
                delta = outputNeuronDelta(neuron, error);
            } else {
                delta = hiddenNeuronDelta(
                        ann,
                        neuron,
                        neuronDeltas,
                        outputError);
            }

            neuronDeltas.insert(&neuron, delta);
            return delta;
        }


        void RpropTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            QHash<const Connection *, qreal> currentGradients;
            QHash<const Connection *, qreal> lastGradients;
            QHash<const Connection *, qreal> updateValues;
            QHash<const Connection *, qreal> lastWeightChange;
            qreal error         = std::numeric_limits<qreal>::max();
            size_t epoch           = 0;

            while (error >= trainingSet.targetError()
                   && ++epoch < trainingSet.maxEpochs()) {
                error = 0.0;
                currentGradients.clear();

                // Forward pass:

                for (const auto &item: trainingSet.trainingData) {
                    Vector errorVector = feedForward(ann, item);
                    error += accumulate(
                            errorVector.begin(),
                            errorVector.end(),
                            0.0,
                            [](const qreal &error, const qreal &delta)
                                    -> qreal {
                                return error + delta * delta;
                            });


                    QHash<const Neuron *, qreal> neuronDeltas;

                    // Calculate error delta of all neurons in the forward pass:

                    ann.eachConnection([
                            this,
                            &ann,
                            &neuronDeltas,
                            &errorVector,
                            &currentGradients ]
                            (const Connection *c) {
                        if (c->fixedWeight()) {
                            return;
                        }

                        const auto &dstNeuron = c->destination();

                        if (ann.inputLayer().contains(dstNeuron)) {
                            return;
                        }

                        qreal delta = neuronDelta(
                                ann,
                                dstNeuron,
                                neuronDeltas,
                                errorVector);
                        neuronDeltas.insert(&dstNeuron, delta);

                        // Add upp gradients. The default-constructed value
                        // for a qreal stored in a QHash is 0.0:

                        currentGradients[c] += delta
                                * c->source().lastResult();
                    });
                }

                // Calculate mean of all errors:

                error /= (trainingSet.trainingData.size()
                        * ann.outputLayer().size());

                // Now, learn:

                ann.eachConnection([
                        this,
                        &currentGradients,
                        &lastGradients,
                        &updateValues,
                        &lastWeightChange ]
                        (Connection *const &c) {
                    if (c->fixedWeight()) {
                        return;
                    }

                    int change = sgn(currentGradients[c] * lastGradients[c]);
                    qreal dw = 0.0;

                    qreal updateValue = DEFAULT_INITIAL_UPDATE;
                    if (updateValues.contains(c)) {
                        updateValue = updateValues[c];
                    }

                    if (0 == change) {
                        qreal delta = updateValue;
                        dw = sgn(currentGradients[c]) * delta;
                        lastGradients[c] = currentGradients[c];
                    } else if (change > 0) { // Retained sign, increase step:
                        qreal delta = updateValue * ETA_POSITIVE;
                        delta = min(delta, MAX_STEP);
                        dw = sgn(currentGradients[c]) * delta;
                        updateValues[c] = delta;
                        lastGradients[c] = currentGradients[c];

                        lastWeightChange[c] = dw;
                    } else { // change < 0 --- Last delta was too big
                        qreal delta = updateValue * ETA_NEGATIVE;
                        delta = max(delta, DELTA_MIN);
                        updateValues[c] = delta;
                        dw = -lastWeightChange[c];

                        // Set the previous gradent to zero so that there will
                        // be no adjustment the next iteration:

                        lastGradients[c] = 0.0;
                    }

                    c->weight(c->weight() + dw);
                });


                LOG4CXX_DEBUG(
                        logger,
                        "Epoch: " << epoch << ", error: " << error);
            }

            setFinalError(trainingSet, error);
            setFinalNumEpochs(trainingSet, epoch);
        }
    } // namespace ANN
} // namespace Winzent
