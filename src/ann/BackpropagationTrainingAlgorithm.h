#ifndef WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H


#include <QObject>
#include <QHash>

#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class Neuron;
        class NeuralNetwork;
        class TrainingSet;


        class BackpropagationTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            /*!
             * The neural network we are currently training.
             */
            NeuralNetwork* m_neuralNetwork;


            /*!
             * Caches the ouptut neuron errors (diffs) of the current epoch.
             */
            ValueVector m_outputError;


            /*!
             * Delta values for the neurons, which are consecutively index
             *
             * \sa NeuralNetwork#m_weightMatrix
             */
            QHash<Neuron*, double> m_deltas;


            /*!
             * Calculates the error per neuron between the actual and the
             * expected output of the neural net.
             */
            ValueVector outputError(
                    const ValueVector &actual,
                    const ValueVector &expected);


            /*!
             * Calculates the neuron delta for one given neuron, assuming it is
             * an output layer neuron.
             */
            double outputNeuronDelta(const Neuron *neuron, const double &error);


            /*!
             * Calculates the delta of a neuron.
             */
            double neuronDelta(Winzent::ANN::Neuron *neuron);


            /*!
             * Calculates the neuron delta for a neuron in an hidden layer.
             */
            double hiddenNeuronDelta(const Neuron *neuron);


            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet);


        public:
            BackpropagationTrainingAlgorithm(QObject *parent = 0);
        };


    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
