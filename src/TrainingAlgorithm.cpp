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


        TrainingAlgorithm::TrainingAlgorithm():
                logger(log4cxx::LogManager::getLogger(
                    "Winzent.ANN.TrainingAlgorithm"))
        {
        }


        TrainingAlgorithm::~TrainingAlgorithm()
        {
        }


        qreal TrainingAlgorithm::calculateMeanSquaredError(
                const Vector &actualOutput,
                const Vector &expectedOutput)
                throw(LayerSizeMismatchException)
        {
            if (actualOutput.size() != expectedOutput.size()) {
                throw LayerSizeMismatchException(
                        actualOutput.size(),
                        expectedOutput.size());
            }


            qreal error = 0.0;
            int n = 0;

            for (auto eit = expectedOutput.begin(),
                        ait = actualOutput.begin();
                    eit != expectedOutput.end() && ait != actualOutput.end();
                    eit++, ait++, n++) {
                error += pow(*eit - *ait, 2);
            }

            error /= static_cast<qreal>(n);
            return error;
        }


        void TrainingAlgorithm::setFinalError(
                TrainingSet &trainingSet,
                const qreal &error)
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
