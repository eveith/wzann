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


namespace Winzent
{
    namespace ANN
    {


        BackpropagationTrainingAlgorithm::BackpropagationTrainingAlgorithm(
                NeuralNetwork *const&network,
                qreal learningRate,
                QObject* parent):
                    TrainingAlgorithm(network, parent),
                    m_outputError(),
                    m_deltas(),
                    m_learningRate(learningRate)
        {
        }


        qreal BackpropagationTrainingAlgorithm::learningRate() const
        {
            return m_learningRate;
        }


        void BackpropagationTrainingAlgorithm::train(
                TrainingSet *const&trainingSet)
        {
            // Initialize the state variables:

            int epochs = 0;
            double error = std::numeric_limits<double>::max();

            // Make sure the network caches the last result; this is important
            // for the process to work.

            setNeuronCacheSize(network(), 1);

            for(; epochs < trainingSet->maxEpochs()
                        && error > trainingSet->targetError();
                    ++epochs) {

                // Begin this run with an error of 0, and clear the state
                // variables:

                error = 0.0;

                // We present each training pattern once per epoch. An epoch
                // is complete once we've presented the complete training set to
                // the network. We then calculate the overall error and try to
                // minimize it.

                QList<TrainingItem>::const_iterator it =
                        trainingSet->trainingData().constBegin();
                for (; it != trainingSet->trainingData().constEnd(); it++) {

                    // Reset memoization fields:

                    m_deltas.clear();
                    m_outputError.clear();

                    // Set up state storage:

                    QHash<Neuron*, double> neuronDeltas;
                    QHash<Connection*, double> connectionDeltas;

                    // First step: Feed forward and compare the network's output
                    // with the ideal teaching output:

                    qDebug() << "Input:" << it->input();
                    ValueVector actualOutput =
                            network()->calculate(it->input());
                    ValueVector expectedOutput = it->expectedOutput();
                    ValueVector errorOutput = outputError(
                            actualOutput,
                            expectedOutput);

                    // Add squared error for this run:

                    foreach (double d, errorOutput) {
                        error += d*d;
                    }

                    // Backpropagation works in two phases: First, the error is
                    // propagated backwarts from the output layer and the
                    // weight deltas are calculated, then the deltas are
                    // applied.

                    for (int i = 0; i != network()->size() - 1; ++i) {
                        Layer *layer = network()->layerAt(i);
                        int layerSize = (network()->inputLayer() == layer
                                ? layer->size()
                                : layer->size() + 1);

                        for (int j = 0; j != layerSize; ++j) {
                            Neuron *neuron = layer->neuronAt(j);

                            QList<Connection *> connections =
                                    network()->neuronConnectionsFrom(neuron);

                            foreach (Connection *c, connections) {
                                double dn = neuronDelta(
                                        c->destination(),
                                        neuronDeltas,
                                        errorOutput);

                                connectionDeltas.insert(
                                    c,
                                    learningRate()
                                    * dn
                                    * c->source()->lastResult());
                            }
                        }
                    }

                    foreach (Connection *c, connectionDeltas.keys()) {
                        qDebug()
                            << c
                            << "w" << c->weight()
                            << "dw" << connectionDeltas[c]
                            << "w" << c->weight() + connectionDeltas[c];
                        c->weight(c->weight() + connectionDeltas[c]);
                    }
                }

                // It's called MEAN square error for a reason:

                error /= (trainingSet->trainingData().count()
                        * trainingSet->trainingData().first()
                            .expectedOutput().count());
                qDebug() << "Error:" << error << " Epochs:" << epochs;
            }

            // Store final training results:

            setFinalError(*trainingSet, error);
            setFinalNumEpochs(*trainingSet, epochs);

            // Restore previously modified cache sizes:

            restoreNeuronCacheSize();
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

            qDebug() << "Output error"
                    << expected << "-" << actual << "=" << error;
            return error;
        }


        double BackpropagationTrainingAlgorithm::outputNeuronDelta(
                const Neuron *neuron,
                const double &error)
                    const
        {
            qDebug() << neuron << "out"
                    << "lastInput" << neuron->lastInput()
                    << "lastResult" << neuron->lastResult();

            return neuron->activationFunction()->calculateDerivative(
                            neuron->lastInput(),
                            neuron->lastResult())
                    * error;
        }


        double BackpropagationTrainingAlgorithm::hiddenNeuronDelta(
                Neuron *neuron,
                QHash<Neuron *, double> &neuronDeltas,
                const ValueVector &outputError)
                    const
        {
            double delta = 0.0;

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


        double BackpropagationTrainingAlgorithm::neuronDelta(
                Neuron *neuron,
                QHash<Neuron *, double> &neuronDeltas,
                const ValueVector &outputError)
                    const
        {
            Q_ASSERT(! network()->inputLayer()->contains(neuron));

            if (neuronDeltas.contains(neuron)) {
                return neuronDeltas[neuron];
            }

            if (network()->outputLayer()->contains(neuron)) {
                int neuronIdx = network()->outputLayer()->indexOf(neuron);
                neuronDeltas.insert(
                        neuron,
                        outputNeuronDelta(neuron, outputError.at(neuronIdx)));
            } else {
                neuronDeltas.insert(
                        neuron,
                        hiddenNeuronDelta(
                                neuron,
                                neuronDeltas,
                                outputError));
            }

            qDebug()
                << neuron
                << (network()->outputLayer()->contains(neuron) ? "out" : "hid")
                << "d" << neuronDeltas[neuron]
                << "(size:" << neuronDeltas.size() << ")";
            return neuronDeltas[neuron];
        }
    }
}
