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


        protected:


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
