#include <cmath>
#include <cstddef>

#include "TrainingSet.h"
#include "NeuralNetwork.h"
#include "LayerSizeMismatchException.h"

#include "TrainingAlgorithm.h"


using std::pow;


namespace wzann {
    TrainingAlgorithm::TrainingAlgorithm()
    {
    }


    TrainingAlgorithm::~TrainingAlgorithm()
    {
    }


    double TrainingAlgorithm::calculateMeanSquaredError(
            Vector const& actualOutput,
            Vector const& expectedOutput)
    {
#ifdef WZANN_DEBUG
        if (actualOutput.size() != expectedOutput.size()) {
            throw LayerSizeMismatchException(
                    actualOutput.size(),
                    expectedOutput.size());
        }
#endif

        double error = 0.0;
        int n = 0;

        for (auto eit = expectedOutput.begin(),
                    ait = actualOutput.begin();
                eit != expectedOutput.end() && ait != actualOutput.end();
                eit++, ait++, n++) {
            error += pow(*eit - *ait, 2);
        }

        error /= static_cast<double>(n);
        return error;
    }


    void TrainingAlgorithm::setFinalError(
            TrainingSet& trainingSet,
            double error)
            const
    {
        trainingSet.m_error = error;
    }


    void TrainingAlgorithm::setFinalNumEpochs(
            TrainingSet& trainingSet,
            size_t epochs)
            const
    {
        trainingSet.m_epochs = epochs;
    }
} // namespace wzann
