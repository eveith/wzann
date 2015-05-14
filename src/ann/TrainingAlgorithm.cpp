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
                log4cxx::LogManager::getLogger(
                    "Winzent.ANN.TrainingAlgorithm");


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
                error += pow(expectedOutput[n] - actualOutput[n], 2);
            }

            error /= static_cast<qreal>(++n);
            return error;
        }


        TrainingAlgorithm::TrainingAlgorithm(QObject *parent):
                QObject(parent)
        {
        }


        void TrainingAlgorithm::setNeuronCacheSize(
                NeuralNetwork *const &network,
                const int &cacheSize)
        {
        }



        void TrainingAlgorithm::restoreNeuronCacheSize()
        {
        }


        void TrainingAlgorithm::setFinalError(
                TrainingSet &trainingSet,
                qreal error)
                const
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
