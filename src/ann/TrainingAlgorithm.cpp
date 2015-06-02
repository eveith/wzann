#include <QtGlobal>

#include <cmath>
#include <cstddef>

#include <log4cxx/logger.h>
#include <log4cxx/logmanager.h>

#include "Exception.h"
#include "TrainingSet.h"
#include "NeuralNetwork.h"

#include "TrainingAlgorithm.h"


using std::pow;


namespace Winzent {
    namespace ANN {


        log4cxx::LoggerPtr TrainingAlgorithm::logger =
                log4cxx::LogManager::getLogger(
                    "Winzent.ANN.TrainingAlgorithm");


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
            int n = 0;

            for (; n != actualOutput.size(); ++n) {
                error += pow(expectedOutput[n] - actualOutput[n], 2);
            }

            error /= static_cast<double>(++n);
            return error;
        }


        void TrainingAlgorithm::setFinalError(
                TrainingSet &trainingSet,
                const double &error)
                const
        {
            trainingSet.m_error = error;
        }


        void TrainingAlgorithm::setFinalNumEpochs(
                TrainingSet &trainingSet,
                const size_t &epochs)
                const
        {
            trainingSet.m_epochs = epochs;
        }
    } // namespace ANN
} // namespace Winzent
