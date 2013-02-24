/*!
 * \file
 */


#include <limits>
#include <cmath>

#include <QHash>

#include "Neuron.h"
#include "Layer.h"
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
                QObject* parent):
                    TrainingAlgorithm(parent),
                    m_neuralNetwork(NULL),
                    m_outputError(ValueVector()),
                    m_deltas(QHash<const Neuron*, double>())
        {
        }


        void BackpropagationTrainingAlgorithm::train(
                NeuralNetwork *network,
                TrainingSet *trainingSet)
        {
            // Initialize the state variables:

            m_neuralNetwork = network;
            int epochs = 0;
            double error = std::numeric_limits<double>::max();

            for(; epochs <= trainingSet->maxEpochs()
                        || error > trainingSet->targetError();
                    ++epochs) {

                // Begin this run with an error of 0, and clear the state
                // variables:

                error = 0.0;
                m_deltas.clear();
                m_outputError.clear();
                m_outputError.fill(0.0, m_neuralNetwork->outputLayer()->size());

                // We present each training pattern once per epoch. An epoch
                // is complete once we've presented the complete training set to
                // the network. We then calculate the overall error and try to
                // minimize it.

                QList<TrainingItem>::const_iterator it =
                        trainingSet->trainingData().constBegin();
                for (; it != trainingSet->trainingData().constEnd(); it++) {

                    // First step: Feed forward and compare the network's output
                    // with the ideal teaching output:

                    ValueVector actualOutput = network->calculate(it->input());
                    ValueVector expectedOutput = it->expectedOutput();
                    m_outputError = outputError(actualOutput, expectedOutput);

                    // Add squared error for this run:

                    foreach (double d, m_outputError) {
                        error += std::pow(d, 2);
                    }

                    // Now backpropagate the error from the output layer to
                    // the input layer.

                    for (int i = m_neuralNetwork->size() - 1; i != 1; --i) {

                    }
                }

                // It's called MEAN square error for a reason:

                error /= (trainingSet->trainingData().count()
                        * trainingSet->trainingData().first()
                            .expectedOutput().count());
            }

            // Store final training results:

            setFinalError(*trainingSet, error);
            setFinalNumEpochs(*trainingSet, epochs);
        }


        ValueVector BackpropagationTrainingAlgorithm::outputError(
                const ValueVector &actual,
                const ValueVector &expected)
        {
            if (actual.size() != expected.size()) {
                throw LayerSizeMismatchException(
                        actual.size(),
                        expected.size());
            }

            ValueVector error;

            for (int i = 0; i != actual.size(); ++i) {
                error[i] = expected[i] - actual[i];
            }

            return error;
        }


        double BackpropagationTrainingAlgorithm::outputNeuronDelta(
                const Neuron *neuron,
                const double &error)
        {
            return neuron->activationFunction()->calculateDerivative(
                    neuron->lastInput()) * error;
        }


        double BackpropagationTrainingAlgorithm::hiddenNeuronDelta(
                const Neuron *neuron)
        {
            QHash<Neuron*, Weight*> connectedNeurons =
                    m_neuralNetwork->connectedNeurons(neuron);
            double delta = 0.0;

            foreach (Neuron *n, connectedNeurons.keys()) {
                // weight(j,k) * delta(k):
                delta += neuronDelta(n) * connectedNeurons[n]->value;
            }

            delta *= neuron->activationFunction()->calculateDerivative(delta);
            return delta;
        }


        double BackpropagationTrainingAlgorithm::neuronDelta(
                const Neuron *neuron)
        {
            // Cast away constness for the list access methods to work...


            Neuron *vneuron = const_cast<Neuron*>(neuron);

            // No training for the input layer:

            if(m_neuralNetwork->inputLayer()->neurons.contains(vneuron)) {
                return 0.0;
            }

            // Memoization: If it is already there, retrieve it from the list.

            if (m_deltas.contains(neuron)) {
                return m_deltas[neuron];
            }

            // If it was not in the list, find out whether it's an output
            // layer neuron or an hidden layer one, and get it's delta.

            if (m_neuralNetwork->outputLayer()->neurons.contains(vneuron)) {
                int neuronIndex =
                        m_neuralNetwork->outputLayer()->
                        neurons.indexOf(vneuron);
                m_deltas[neuron] = outputNeuronDelta(
                        neuron,
                        m_outputError[neuronIndex]);
            } else {
                m_deltas[neuron] = hiddenNeuronDelta(neuron);
            }

            return m_deltas[neuron];
        }
    }
}
