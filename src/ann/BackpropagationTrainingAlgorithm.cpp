/*!
 * \file
 */


#include <limits>

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
                    m_deltas(QHash<const Neuron*, double>())
        {
        }


        void BackpropagationTrainingAlgorithm::train(
                NeuralNetwork *network,
                TrainingSet *trainingSet)
        {
            // Initialize the state variables:

            m_neuralNetwork = network;
            m_deltas.clear();
            int epochs = 0;
            double error = std::numeric_limits<double>::max();

            for(; epochs <= trainingSet->maxEpochs()
                        || error > trainingSet->targetError();
                    ++epochs) {

                // Begin this run with an error of 0:

                error = 0.0;

                // We present each training pattern once per epoch. An epoch
                // is complete once we've presented the complete training set to
                // the network. We then calculate the overall error and try to
                // minimize it.

                QList<TrainingItem>::const_iterator it =
                        trainingSet->trainingData().constBegin();
                for (; it != trainingSet->trainingData().constEnd(); it++) {
                    ValueVector actualOutput = network->calculate(it->input());
                    ValueVector expectedOutput = it->expectedOutput();
                    ValueVector outputErrorVector = outputError(
                            actualOutput,
                            expectedOutput);

                    // Add squared error for this run, and meanwhile calculate
                    // the output delta.

                    for (int i = 0; i != outputErrorVector.size(); ++i) {
                        error += outputErrorVector[i];
                    }
                }

                // It's called MEAN square error for a reason:

                error /= trainingSet->trainingData().count();
            }

            // Store final training results:

            setFinalError(*trainingSet, error);
            setFinalNumEpochs(*trainingSet, epochs);
#if 0
    int numIterations = 0;
    QVector<TrainingItem>::const_iterator it =
            trainingSet.m_trainingData.constBegin();
    double error = INFINITY;

    while (numIterations < trainingSet.m_maxNumEpochs
            && error / (numIterations+1) > trainingSet.m_targetError) {

        // Wrap iterator if we've reached the end, and increase number of
        // iterations:

        if (trainingSet.m_trainingData.constEnd() == it) {
            it = trainingSet.m_trainingData.constBegin();
            ++numIterations;
        }

        const ValueVector input(it->input());
        const ValueVector expectedOutput(it->expectedOutput());

        // First step: feed-forward the current training input:

        const ValueVector actualOutput = network->calculate(input);

        // Second step: Calculate output layer error and also update the
        // overall error in the training set:

        error = this->calculateMeanSquaredError(
                    actualOutput,
                    expectedOutput);
        QVector<double> outputError(m_numNeuronsPerLayer[INDEX_OUTPUT]);

        for (int i = 0; i != actualOutput.size(); ++i) {
            outputError[i] = (expectedOutput[i] - actualOutput[i])
                    * m_activationFunctionOutput->calculateDerivative(
                            actualOutput[i]);

            error += std::pow(expectedOutput[i] - actualOutput[i], 2);
        }
        error *= 0.5;

        // Third step: Calculate hidden layer error:

        QVector<double> hiddenError(m_numNeuronsPerLayer[INDEX_HIDDEN]);

        int* idxHidden = matrixIndices(INDEX_HIDDEN);
        int* idxOutput = matrixIndices(INDEX_OUTPUT);

        for (int i = idxHidden[0]; i <= idxHidden[1]; ++i){
            int k = i - idxHidden[0];
            hiddenError[k] = 0.0;

            for(int j = idxOutput[0]; j <= idxOutput[1]; ++j) {
                int l = j - idxOutput[0];

                if(NULL != m_weightMatrix[i][j]) {
                    hiddenError[k] += outputError[l] * *m_weightMatrix[i][j];
                }
            }
            hiddenError[k] *= m_activationFunctionHidden->calculateDerivative(
                    m_lastOutputs[i]);
        }

        // Fourth step: Update weights to output layer neurons:

        for (int i = idxOutput[0]; i <= idxOutput[1]; ++i) {
            int k = i - idxOutput[0];

            for (int j = idxHidden[0]; j <= idxHidden[1]; ++j) {
                if (NULL != m_weightMatrix[j][i]) {
                    *m_weightMatrix[j][i] += (trainingSet.m_learningRate
                            * outputError[k]
                            * m_lastOutputs[j]);
                }
            }
        }

        // Fifth step: Update weights to hidden layer neurons:

        int* idxInput = matrixIndices(INDEX_INPUT);

        for (int i = idxHidden[0]; i <= idxHidden[1]; ++i) {
            int k = i - idxHidden[0];

            for (int j = idxInput[0]; j <= idxInput[1]; ++j) {
                int l = j - idxInput[0];

                if (NULL != m_weightMatrix[j][i]) {
                    *m_weightMatrix[j][i] += (trainingSet.m_learningRate
                            * hiddenError[k]
                            * input[l]);
                }
            }
        }

        it++;

        delete idxInput;
        delete idxHidden;
        delete idxOutput;
    }

    trainingSet.m_epochs = numIterations;
    trainingSet.m_error = error / (numIterations+1);
    return trainingSet.error();
#endif
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
            // Memoization: If it is already there, retrieve it from the list.

            if (m_deltas.contains(neuron)) {
                return m_deltas[neuron];
            }

            // If it was not in the list, find out whether it's an output
            // layer neuron or an hidden layer one, and get it's delta.

            if (m_neuralNetwork->outputLayer()->neurons.contains(
                    const_cast<Neuron*>(neuron))) {
                m_deltas[neuron] = outputNeuronDelta(neuron, 0.0);
            } else {
                m_deltas[neuron] = hiddenNeuronDelta(neuron);
            }

            return m_deltas[neuron];
        }
    }
}
