#include <cmath>

#include "Exception.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "TrainingSet.h"

#include "TrainingAlgorithm.h"


using std::pow;


namespace Winzent
{
    namespace ANN
    {
        double TrainingAlgorithm::calculateMeanSquaredError(
                const ValueVector &actualOutput,
                const ValueVector &expectedOutput)
                    throw(LayerSizeMismatchException)
        {
            if (actualOutput.size() != expectedOutput.size()) {
                throw LayerSizeMismatchException(
                            actualOutput.size(),
                            expectedOutput.size());
            }


            double error = 0.0;

            for (int i = 0; i != actualOutput.size(); ++i) {
                error += pow((actualOutput[i] - expectedOutput[i]), 2);
            }

            error /= static_cast<double>(actualOutput.size());
            return error;
        }


        TrainingAlgorithm::TrainingAlgorithm(QObject *parent) :
                QObject(parent),
                m_cacheSizes(QHash<Neuron *, int>())
        {
        }


        void TrainingAlgorithm::setNeuronCacheSize(
                NeuralNetwork *network,
                int cacheSize)
        {
            for (int i = 0; i != network->size(); ++i) {
                Layer *layer = network->layerAt(i);
                for (int j = 0; j != layer->size(); ++j) {
                    Neuron *neuron = layer->neuronAt(j);
                    m_cacheSizes.insert(neuron, neuron->cacheSize());
                    neuron->cacheSize(cacheSize);
                }

                // Make sure the bias neuron is also saved:

                m_cacheSizes.insert(
                        layer->biasNeuron(),
                        layer->biasNeuron()->cacheSize());
                layer->biasNeuron()->cacheSize(cacheSize);
            }
        }



        void TrainingAlgorithm::restoreNeuronCacheSize()
        {
            foreach (Neuron *neuron, m_cacheSizes.keys()) {
                neuron->cacheSize(m_cacheSizes[neuron]);
            }

            m_cacheSizes.clear();
        }


        void TrainingAlgorithm::setFinalError(
                TrainingSet &trainingSet,
                double error) const
        {
            trainingSet.m_error = error;
        }


        void TrainingAlgorithm::setFinalNumEpochs(
                TrainingSet &trainingSet,
                int epochs) const
        {
            trainingSet.m_epochs = epochs;
        }
    } // namespace ANN
} // namespace Winzent
