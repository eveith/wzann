#ifndef TRAININGSET_H_
#define TRAININGSET_H_


#include <QtGlobal>
#include <QList>

#include <cstddef>
#include <ostream>

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
            Vector m_input;


            /*!
             * The output expected from the network.
             */
            Vector m_expectedOutput;


            /*!
             * \brief Stores whether an expected output to the input exists
             *  (i.e., the output is relevant) or not.
             */
            bool m_outputRelevant;


        public:


            /*!
             * Constructs a new instance given input and expected
             * output.
             */
            TrainingItem(
                    const Vector &input,
                    const Vector &expectedOutput);


            /*!
             * Constructs a new training item without an expected output: This
             * Items is fed to the network during training, but its output is
             * discarded and not added in during error calculation. Useful for
             * recurrent networks.
             *
             * \param[in] input The input that is fed to the neural network.
             *
             * \sa #outputRelevant
             */
            TrainingItem(const Vector &input);


            /*!
             * \brief Constructs a new, empty training item.
             */
            TrainingItem()
            {
            }


            /*!
             * Creates a copy of the other training item.
             */
            TrainingItem(const TrainingItem &rhs);


            /*!
             * \return The input for the net
             */
            const Vector input() const;


            /*!
             * \return The output that is expected of the net
             */
            const Vector expectedOutput() const;


            /*!
             * \return Whether an expected output exists and is relevant or not
             */
            bool outputRelevant() const;


            /*!
             * Calculates the error of each item in a vector. This method
             * applies the simple formula `error[i] = expected[i]-actual[i]`.
             *
             * \param[in] actualOutput The output the network actually emitted
             *
             * \return The error between the output the network emitted and that
             *  which is stored in the training item.
             *
             * \throw LayerSizeMismatchException If the two vectors
             *  expectedOutput and actualOutput do not match; throws this also
             *  if the output is not relevant for this training item.
             */
            Vector error(const Vector &actualOutput) const
                    throw(LayerSizeMismatchException);
        };


        /*!
         * \brief Represents a complete set of training data.
         *
         * A set of training data is a collection of inputs together with
         * their desired outputs. The training set is used to train the
         * network until either the target error or the maximum number
         * of iteration have been reached. The resulting target error
         * is then stored in the particular training set instance.
         */
        class TrainingSet
        {
            friend class TrainingAlgorithm;


        public:


            //! A vector or list of training items
            typedef QList<TrainingItem> TrainingItems;


            /*!
             * Constructs a new TrainingSet by supplying the training
             * data and the relevant paramters.
             *
             * \param[in] trainingData The data used for training, given
             *  as a Hash mapping input to output. Both are
             *  <code>qreal[]</code>, where each array index
             *  corresponds to a neuron.
             *
             * \param[in] targetError The target mean square error after
             *  which the training stops
             *
             * \param[in] maxNumEpochs The maximum number of epochs the
             *  training runs for. If this number is reached the
             *  training will end, even if the target error is not
             *  yet reached.
             */
            TrainingSet(
                    TrainingItems trainingData,
                    const qreal &targetError,
                    const size_t &maxNumEpochs);


            /*!
             * Returns the mean square error after the current
             * training epoch.
             */
            qreal error() const;


            /*!
             * Returns the previously set target error.
             *
             * \return The target error.
             */
            qreal targetError() const;


            /*!
             * Returns the maximum number of epochs that are allowed during
             * training.
             */
            size_t maxEpochs() const;


            /*!
             * Returns the number of epochs needed to complete
             * the training.
             */
            size_t epochs() const;


            /*!
             * \brief Allows const access to all training items
             *
             * \return All training items.
             */
            TrainingItems trainingData() const;


            /*!
             * \brief Adds a copy of the given TrainingItem to the list of
             *  training items
             *
             * \param[in] item The new item
             *
             * \return `*this`
             */
            TrainingSet &operator <<(const TrainingItem &item);


            /*!
             * \brief Allows read-write access to the training item at
             *  a certain index
             *
             * \param[in] index Index of the training item
             *
             * \return The Training Item, modifiable
             */
            TrainingItem &operator[](const size_t &index);


        private:


            /*!
             * The data we're training on.
             */
            TrainingItems m_trainingData;


            /*!
             * The error we're trying to target.
             */
            qreal m_targetError;


            /*!
             * Maximum number of epochs the training will run for.
             */
            size_t m_maxNumEpochs;


            /*!
             * Number of epochs it took to complete the training.
             */
            size_t m_epochs;


            /*!
             * The error after each epoch (stores the
             * final error after the training is finished.)
             */
            qreal m_error;

        };
    } /* namespace ANN */
} /* namespace Winzent */


namespace std {
    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingItem &trainingItem);
    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingSet::TrainingItems &trainingData);
    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingSet &trainingSet);
}

#endif /* TRAININGSET_H_ */
