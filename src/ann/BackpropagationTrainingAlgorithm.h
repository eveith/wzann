#ifndef WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H


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
        private:


            /*!
             * The neural network we are currently training.
             */
            NeuralNetwork *m_neuralNetwork;


            /*!
             * Caches the ouptut neuron errors (diffs) of the current epoch.
             */
            ValueVector m_outputError;


            /*!
             * Delta values for the neurons, which are consecutively index
             *
             * \sa NeuralNetwork#m_weightMatrix
             */
            QHash<Neuron *, double> m_deltas;


            /*!
             * The learning rate applied to each weight change
             */
            double m_learningRate;


            /*!
             * Calculates the error per neuron between the actual and the
             * expected output of the neural net.
             */
            ValueVector outputError(
                    const ValueVector &actual,
                    const ValueVector &expected)
                        const;


            /*!
             * \brief Calculates the neuron delta for one given neuron,
             *  assuming it is an output layer neuron.
             */
            double outputNeuronDelta(const Neuron &neuron, const double &error)
                    const;


            /*!
             * Calculates the neuron delta for a neuron in an hidden layer.
             */
            double hiddenNeuronDelta(
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, double> &neuronDeltas,
                    const ValueVector &outputError)
                        const;


            /*!
             * Calculates the delta value of a neuron.
             */
            double neuronDelta(
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, double> &neuronDeltas,
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
            BackpropagationTrainingAlgorithm(double learningRate = 0.7);


            /*!
             * \return The learning rate applied to each weight change
             */
            double learningRate() const;


            /*!
             * \brief Sets the learning rate of the Backpropagation training
             *  algorithm
             *
             * \param[in] rate The new rate
             *
             * \return `*this`
             */
            BackpropagationTrainingAlgorithm &learningRate(const double &rate);


            /*!
             * \brief Trains the neural network with the given trainingSet.
             *
             * \param[in] trainingSet A set of sample inputs and expected
             *  outputs.
             */
            virtual void train(NeuralNetwork &ann, TrainingSet &trainingSet)
                    override;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_BACKPROPAGATIONTRAININGALGORITHM_H
