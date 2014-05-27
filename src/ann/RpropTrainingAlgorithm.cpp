#include <QObject>

#include <limits>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "Layer.h"

#include "Neuron.h"
#include "ActivationFunction.h"

#include "TrainingAlgorithm.h"
#include "RpropTrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {



        const qreal RpropTrainingAlgorithm::ETA_POSITIVE =  1.2;
        const qreal RpropTrainingAlgorithm::ETA_NEGATIVE = -0.5;
        const qreal RpropTrainingAlgorithm::ZERO_TOLERANCE =0.00000000000000001;



        RpropTrainingAlgorithm::RpropTrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent):
                    TrainingAlgorithm(network, parent)
        {
        }


        int RpropTrainingAlgorithm::sgn(const qreal &x)
        {
            if (std::fabs(x) < ZERO_TOLERANCE) {
                return 0;
            }

            if (x < 0.0) {
                return -1;
            }

            return 1;
        }


        ValueVector RpropTrainingAlgorithm::outputError(
                const ValueVector &expected,
                const ValueVector &actual)
                const
        {
            Q_ASSERT(expected.size() == actual.size());

            ValueVector error;
            error.reserve(actual.size());

            for (int i = 0; i != actual.size(); ++i) {
                error << expected.at(i) - actual.at(i);
            }

            return error;
        }


        qreal RpropTrainingAlgorithm::outputNeuronDelta(
                const Neuron *const &neuron,
                const qreal &error)
                const
        {
            return error * neuron->activationFunction()->calculateDerivative(
                    neuron->lastInput(),
                    neuron->lastResult());
        }


        qreal RpropTrainingAlgorithm::hiddenNeuronDelta(
                Neuron *const &neuron,
                QHash<Neuron *, qreal> &neuronDeltas,
                const ValueVector &outputError)
                const
        {
            qreal delta = 0.0;

            QList<Connection*> connections =
                    network()->neuronConnectionsFrom(neuron);
            Q_ASSERT(connections.size() > 0);

            foreach (Connection *c, connections) {
                // weight(j,k) * delta(k):
                delta += neuronDelta(
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


        qreal RpropTrainingAlgorithm::neuronDelta(
                const Neuron *const &neuron,
                QHash<Neuron *, qreal> &neuronDeltas,
                const ValueVector &outputError)
                    const
        {
            // Apply memoization, where possible:

            if (neuronDeltas.contains(neuron)) {
                return neuronDeltas.value(neuron);
            }

            // What layer does the neuron live in?

            Q_ASSERT(!network()->inputLayer()->contains(neuron));
            qreal delta = 0.0;

            if (network()->outputLayer()->contains(neuron)) {
                qreal error = outputError.at(
                        network()->outputLayer()->indexOf(neuron));
                delta = outputNeuronDelta(neuron, error);
            } else {
                delta = hiddenNeuronDelta(neuron, neuronDeltas, outputError);
            }

            neuronDeltas.insert(neuron, delta);
            return delta;
        }


        void RpropTrainingAlgorithm::train(TrainingSet *const &trainingSet)
        {
            qreal error         = std::numeric_limits<qreal>::max();
            qreal lastError     = std::numeric_limits<qreal>::max();
            qreal delta         = 0;
            qreal lastDelta     = 0;
            qreal gradient      = 0;
            qreal lastGradient  = 0;
            int epoch           = 0;

            setNeuronCacheSize(network(), 2);

            while (error >= trainingSet->targetError()
                   && ++epoch < trainingSet->maxEpochs()) {

                // Forward pass:

                lastError = error;
                error = 0;

                foreach (TrainingItem item, trainingSet->trainingData()) {
                    ValueVector actual   = network()->calculate(item.input());
                    ValueVector expected = item.expectedOutput();
                    ValueVector outputError = outputError(expected, actual);

                    error += calculateMeanSquaredError(actual, expected);
                }

                error /= trainingSet->trainingData().size();
            }

            setFinalError(*trainingSet, error);
            setFinalNumEpochs(*trainingSet, epoch);
        }
    } // namespace ANN
} // namespace Winzent
