/*
 * TrainingSet.h
 *
 *  Created on: 18.10.2012
 *      Author: eveith
 */

#ifndef TRAININGSET_H_
#define TRAININGSET_H_


#include <QObject>
#include <QList>

#include "NeuralNetwork.h"


namespace Winzent {
    namespace ANN {


        class TrainingAlgorithm;


        /*!
         * \brief Represents one training input and expected output
         *
         * This class represents one item a net can train on. It
         * contains the input values and the expected output values.
         * It also contains a method to calculate the RMSE if an
         * actual output is given.
         *
         * \sa #rootMeanSquareError
         */
        class TrainingItem
        {

        private:


            /*!
             * The input presented to the network
             */
            ValueVector m_input;


            /*!
             * The output expected from the network.
             */
            ValueVector m_expectedOutput;


        public:


            /*!
             * Constructs a new instance given input and expected
             * output.
             */
            TrainingItem(
                    ValueVector input,
                    ValueVector expectedOutput);


            /*!
             * Constructs a new, empty training item.
             */
            TrainingItem():
                    m_input(ValueVector()),
                    m_expectedOutput(ValueVector())
            {
            }


            /*!
             * Creates a copy of the other training item.
             */
            TrainingItem(const TrainingItem& copy);


            /*!
             * \return The input for the net
             */
            const ValueVector input() const {
                return m_input;
            }


            /*!
             * \return The output that is expected of the net
             */
            const ValueVector expectedOutput() const {
                return m_expectedOutput;
            }
        };


        /*!
         * Represents a complete set of training data. A set of
         * training data is a collection of inputs together with their
         * desired outputs. The training set is used to train the
         * network until either the target error or the maximum number
         * of iteration have been reached. The resulting target error
         * is then stored in the particular training set instance.
         */
        class TrainingSet: public QObject
        {
            Q_OBJECT


            friend class TrainingAlgorithm;


        private:


            /*!
             * The data we're training on.
             */
            QList<TrainingItem> m_trainingData;


            /*!
             * The learning rate applied to the net.
             */
            double m_learningRate;


            /*!
             * The error we're trying to target.
             */
            double m_targetError;


            /*!
             * Maximum number of epochs the training will run for.
             */
            int m_maxNumEpochs;


            /*!
             * Number of epochs it took to complete the training.
             */
            int m_epochs;


            /*!
             * The error after each epoch (stores the
             * final error after the training is finished.)
             */
            double m_error;


        public:


            /*!
             * Constructs a new TrainingSet by supplying the training
             * data and the relevant paramters.
             *
             * \param trainingData The data used for training, given
             *  as a Hash mapping input to output. Both are
             *  <code>double[]</code>, where each array index
             *  corresponds to a neuron.
             *
             * \param learningRate The learning rate applied to the
             *  net
             *
             * \param targetMSE The target mean square error after
             *  which the training stops
             *
             * \param maxNumEpochs The maximum number of epochs the
             *  training runs for. If this number is reached the
             *  training will end, even if the target error is not
             *  yet reached.
             */
            TrainingSet(
                    QList<TrainingItem> trainingData,
                    double learningRate,
                    double targetError,
                    int maxNumEpochs);


            virtual ~TrainingSet();


            /*!
             * Returns the mean square error after the current
             * training epoch.
             */
            double error() const;


            /*!
             * Returns the previously set target error.
             *
             * \return The target error.
             */
            double targetError() const;


            /*!
             * Returns the maximum number of epochs that are allowed during
             * training.
             */
            int maxEpochs() const;


            /*!
             * Returns the number of epochs needed to complete
             * the training.
             */
            int epochs() const;


            /*!
             * Returns the learning rate that should be applied.
             */
            double learningRate() const;


            const QList<TrainingItem> trainingData() const;
        };

    } /* namespace ANN */
} /* namespace Winzent */
#endif /* TRAININGSET_H_ */
