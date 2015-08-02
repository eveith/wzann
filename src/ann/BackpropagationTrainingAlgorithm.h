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
            Vector m_outputError;


            /*!
             * Delta values for the neurons, which are consecutively index
             *
             * \sa NeuralNetwork#m_weightMatrix
             */
            QHash<Neuron *, qreal> m_deltas;


            /*!
             * The learning rate applied to each weight change
             */
            qreal m_learningRate;


            /*!
             * Calculates the error per neuron between the actual and the
             * expected output of the neural net.
             */
            Vector outputError(
                    const Vector &actual,
                    const Vector &expected)
                        const;


            /*!
             * \brief Calculates the neuron delta for one given neuron,
             *  assuming it is an output layer neuron.
             */
            qreal outputNeuronDelta(const Neuron &neuron, const qreal &error)
                    const;


            /*!
             * Calculates the neuron delta for a neuron in an hidden layer.
             */
            qreal hiddenNeuronDelta(
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, qreal> &neuronDeltas,
                    const Vector &outputError)
                        const;


            /*!
             * Calculates the delta value of a neuron.
             */
            qreal neuronDelta(
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, qreal> &neuronDeltas,
                    const Vector &outputError)
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
            BackpropagationTrainingAlgorithm(qreal learningRate = 0.7);


            /*!
             * \return The learning rate applied to each weight change
             */
            qreal learningRate() const;


            /*!
             * \brief Sets the learning rate of the Backpropagation training
             *  algorithm
             *
             * \param[in] rate The new rate
             *
             * \return `*this`
             */
            BackpropagationTrainingAlgorithm &learningRate(const qreal &rate);


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
