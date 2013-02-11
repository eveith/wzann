#include <cmath>

#include "Exception.h"
#include "NeuralNetwork.h"
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
            QObject(parent)
        {
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
