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
             * The learning rate applied to each weight change
             */
            qreal m_learningRate;


            /*!
             * Calculates the error per neuron between the actual and the
             * expected output of the neural net.
             */
            ValueVector outputError(
                    const ValueVector &actual,
                    const ValueVector &expected)
                        const;


            /*!
             * Calculates the neuron delta for one given neuron, assuming it is
             * an output layer neuron.
             */
            double outputNeuronDelta(const Neuron *neuron, const double &error)
                    const;


            /*!
             * Calculates the neuron delta for a neuron in an hidden layer.
             */
            double hiddenNeuronDelta(
                    Neuron *neuron,
                    QHash<Neuron *, double> &neuronDeltas,
                    const ValueVector &outputError)
                        const;


            /*!
             * Calculates the delta value of a neuron.
             */
            double neuronDelta(
                    Neuron *neuron,
                    QHash<Neuron *, double> &neuronDeltas,
                    const ValueVector &outputError)
                        const;


        public:


            /*!
             * Constructs a new instance of the Backpropagation training
             * algorithm.
             *
             * \param[in] network The network to be trained
             *
             * \param[in] learningRate The learning rate applied to each weight
             *  change: Literature suggest `0.7` as a sensible starting value.
             *
             * \param parent The parent object
             */
            BackpropagationTrainingAlgorithm(
                    NeuralNetwork *const &network,
                    qreal learningRate,
                    QObject *parent = 0);


            /*!
             * \return The learning rate applied to each weight change
             */
            inline qreal learningRate() const;


            virtual void train(TrainingSet *const &trainingSet);
        };


    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
