#include <cmath>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include "Exception.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "TrainingSet.h"

#include "TrainingAlgorithm.h"


using std::pow;


namespace Winzent {
    namespace ANN {


        log4cxx::LoggerPtr TrainingAlgorithm::logger =
                log4cxx::LogManager::getLogger("Winzent.ANN.TrainingAlgorithm");


        qreal TrainingAlgorithm::calculateMeanSquaredError(
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
            int n = 0;

            for (; n != actualOutput.size(); ++n) {
                error += pow((actualOutput[n] - expectedOutput[n]), 2);
            }

            error /= static_cast<qreal>(n);
            return error;
        }


        TrainingAlgorithm::TrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent) :
                    QObject(parent),
                    m_cacheSizes(),
                    m_network(network)
        {
            if (0 == parent) {
                setParent(network);
            }
        }


        void TrainingAlgorithm::setNeuronCacheSize(
                NeuralNetwork *const&network,
                const int &cacheSize)
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


        NeuralNetwork *TrainingAlgorithm::network() const
        {
            return m_network;
        }
    } // namespace ANN
} // namespace Winzent
