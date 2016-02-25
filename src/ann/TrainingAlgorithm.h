/*!
 * \file TrainingAlgorithm.h
 */


#ifndef WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H
#define WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H



#include <cstddef>

#include <log4cxx/logger.h>

#include "Exception.h"
#include "NeuralNetwork.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class TrainingSet;


        /*!
         * \brief Abstract training algorithm interface for all neural
         *  network training algorithms.
         */
        class TrainingAlgorithm
        {
        public:


            /*!
             * \brief Calculates the mean square error.
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
            static qreal calculateMeanSquaredError(
                    const Vector &actualOutput,
                    const Vector &expectedOutput)
                    throw(LayerSizeMismatchException);


            TrainingAlgorithm();
            virtual ~TrainingAlgorithm();


            /*!
             * \brief Commences the training of the neural network.
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
             * \param ann The artificial neural network we want to train
             *
             * \param trainingSet A training set supplying training
             *  data and other information
             *
             * \sa TrainingSet
             *
             * \sa NeuralNetwork#train
             */
            virtual void train(
                    NeuralNetwork &neuralNetwork,
                    TrainingSet &trainingSet) = 0;


        protected:


            //! \brief Internal logger
            log4cxx::LoggerPtr logger;


            /*!
             * \brief Sets the final error of a training set.
             *
             * \param[inout] trainingSet The training set on which to record
             *  the final error
             *
             * \param[in] error The final error
             */
            void setFinalError(TrainingSet &trainingSet, const qreal &error)
                    const;


            /*!
             * \brief Sets the final number of epochs needed for the training.
             *
             * \param[inout] trainingSet The training set on which to set
             *  the final number of epochs
             *
             * \param[in] epochs The number of epochs the training took.
             */
            void setFinalNumEpochs(
                    TrainingSet &trainingSet,
                    const size_t &epochs)
                    const;

        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_MODEL_FORECASTER_ANN_TRAININGALGORITHM_H
