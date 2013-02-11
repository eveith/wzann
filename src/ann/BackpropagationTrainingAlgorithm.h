#ifndef WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H


#include <QObject>

#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class Neuron;
        class TrainingSet;


        class BackpropagationTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            /*!
             * Calculates the error per neuron between the actual and the
             * expected output of the neural net.
             */
            ValueVector outputError(
                    const ValueVector &actual,
                    const ValueVector &expected);


            /*!
             * Calculates the neuron delta for one given neuron.
             */
            double neuronDelta(const Neuron *neuron, const double &error);


            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet);


        public:
            BackpropagationTrainingAlgorithm(QObject *parent = 0);
        };


    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
