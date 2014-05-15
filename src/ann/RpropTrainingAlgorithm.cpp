#include <QObject>

#include <limits>

#include "NeuralNetwork.h"
#include "TrainingSet.h"

#include "TrainingAlgorithm.h"
#include "RpropTrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {



        const qreal RpropTrainingAlgorithm::ETA_POSITIVE =  1.2;
        const qreal RpropTrainingAlgorithm::ETA_NEGATIVE = -0.5;



        RpropTrainingAlgorithm::RpropTrainingAlgorithm(
                NeuralNetwork *const &network,
                QObject *parent):
                    TrainingAlgorithm(network, parent)
        {
        }


        void RpropTrainingAlgorithm::train(TrainingSet *const &trainingSet)
        {
            qreal error         = std::numeric_limits<qreal>::max();
            qreal lastError     = std::numeric_limits<qreal>::max();
            qreal delta         = 0;
            qreal lastDelta     = 0;
            qreal gradient      = 0;
            qreal lastGradient  = 0;
            int epoch           = 0;

            setNeuronCacheSize(network(), 2);

            while (error >= trainingSet->targetError()
                   && ++epoch < trainingSet->maxEpochs()) {

                // Forward pass:

                lastError = error;
                error = 0;

                foreach (TrainingItem item, trainingSet->trainingData()) {
                    ValueVector actual   = network()->calculate(item.input());
                    ValueVector expected = item.expectedOutput();

                    error += calculateMeanSquaredError(actual, expected);
                }

                error /= trainingSet->trainingData().size();
            }

            setFinalError(*trainingSet, error);
            setFinalNumEpochs(*trainingSet, epoch);
        }
    } // namespace ANN
} // namespace Winzent
