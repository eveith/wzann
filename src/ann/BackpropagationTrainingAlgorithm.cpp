/*!
 * \file
 */


#include "NeuralNetwork.h"
#include "TrainingSet.h"
#include "BackpropagationTrainingAlgorithm.h"


namespace Winzent
{
    namespace ANN
    {


        BackpropagationTrainingAlgorithm::BackpropagationTrainingAlgorithm(
                QObject* parent): TrainingAlgorithm(parent)
        {
        }


        void BackpropagationTrainingAlgorithm::train(
                NeuralNetwork *network,
                TrainingSet *trainingSet)
        {
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
    }
}
