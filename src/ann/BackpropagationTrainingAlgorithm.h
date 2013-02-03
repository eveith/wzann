#ifndef WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H


#include <QObject>
#include "TrainingAlgorithm.h"

namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class TrainingSet;


        class BackpropagationTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet);


        public:
            BackpropagationTrainingAlgorithm(QObject *parent = 0);
        };


    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
