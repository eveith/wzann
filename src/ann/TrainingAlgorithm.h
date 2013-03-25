/*!
 * \file TrainingAlgorithm.h
 */


#ifndef WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H


#include <QObject>

#include "Exception.h"
#include "NeuralNetwork.h"


namespace Winzent
{
    namespace ANN
    {


        class NeuralNetwork;
        class TrainingSet;


        /*!
         * Abstract training algorithm interface for all neural
         * network training algorithms.
         */
        class TrainingAlgorithm: public QObject
        {
            Q_OBJECT


            friend void NeuralNetwork::train(
                    TrainingAlgorithm*,
                    TrainingSet*);


        private:


            /*!
             * Used for storing the current cache size of all network neurons.
             *
             * \sa #setNeuronCacheSize
             *
             * \sa #restoreNeuronCacheSize
             */
            QHash<Neuron *, int> m_cacheSizes;


        protected:


            /*!
             * Sets the final error of a training set.
             */
            void setFinalError(TrainingSet &trainingSet, double error) const;


            /*!
             * Sets the final number of epochs needed for the training.
             */
            void setFinalNumEpochs(TrainingSet &trainingSet, int epochs) const;


            /*!
             * Calculates the mean square error.
             *
             * The MSE is defined as the sum of all squared errors,
             * divided by the number of output neurons.
             *
             * \param actualOutput The actual output the network
             *  generated
             *
             * \param expectedOutput The output that was expected from
             *  the network
             *
             * \return The mean squared error
             *
             * \throws LayerSizeMismatchException if the number of
             *  neurons in one input vector differs from the other.
             */
            double calculateMeanSquaredError(
                    const ValueVector &actualOutput,
                    const ValueVector &expectedOutput)
                        throw(LayerSizeMismatchException);


            /*!
             * Sets the cache size of all neurons in the network to a certain
             * value. The current cache sizes are stored and can be reset using
             * #restoreNeuronCacheSize.
             *
             * \param network The network that contains the neurons.
             *
             * \param cacheSize The cache size
             *
             * \sa #restoreNeuronCacheSize
             */
            void setNeuronCacheSize(NeuralNetwork *network, int cacheSize);


            /*!
             * Restores the cache sizes of all network neurons.
             *
             * \sa #setNeuronCacheSize
             */
            void restoreNeuronCacheSize();


            /*!
             * Commences the training of the neural network.
             *
             * How this training is being done is up to the training
             * strategy. The <code>TrainingSet</code> supplies a number
             * of information that can be helpful to extract a stop
             * condition, e.g. a maximum number of iterations.
             *
             * This method is private, but can be accessed from the
             * <code>NeuralNetwork</code> class instance. This way, the
             * internals of the calling interface stay hidden from the
             * outside world.
             *
             * \param network The network that shall be trained
             *
             * \param trainingSet A training set supplying training
             *  data and other information
             *
             * \sa TrainingSet
             *
             * \sa NeuralNetwork#train
             */
            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet) = 0;


        public:


            explicit TrainingAlgorithm(QObject *parent = 0);
        };

    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H
