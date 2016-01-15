#ifndef TRAININGSET_H_
#define TRAININGSET_H_


#include <QtGlobal>
#include <QList>

#include <cstddef>
#include <ostream>

#include "Vector.h"
#include "NeuralNetwork.h"

#include <JsonSerializable.h>


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
        class WINZENTANNSHARED_EXPORT TrainingItem: public JsonSerializable
        {
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
            explicit TrainingItem()
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
            Vector error(const Vector &actualOutput) const;


            //! \brief Clears the TrainingItem
            virtual void clear() override;


            /*!
             * \brief Serializes the TrainingItem to JSON
             *
             * \return The JSON representation of the TrainingItem
             */
            virtual QJsonDocument toJSON() const override;


            /*!
             * \brief Reinstates a serialized JSON TrainingItem
             *
             * \param[in] json The TrainingItem's JSON representation
             */
            virtual void fromJSON(const QJsonDocument &json) override;


        private:


            //! \brief The input presented to the network
            Vector m_input;


            //! \brief The output expected from the network.
            Vector m_expectedOutput;


            /*!
             * \brief Stores whether an expected output to the input exists
             *  (i.e., the output is relevant) or not.
             */
            bool m_outputRelevant;

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
        class WINZENTANNSHARED_EXPORT TrainingSet: public JsonSerializable
        {
            friend class TrainingAlgorithm;


        public:


            //! A vector or list of training items
            typedef QList<TrainingItem> TrainingItems;


            //! The actual set of training data
            TrainingItems trainingData;


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


            //! Creates an empty training set.
            explicit TrainingSet();


            /*!
             * \brief Returns the mean square error after the current
             *  training epoch.
             *
             * \return The actual, current error
             */
            qreal error() const;


            /*!
             * \brief Returns the previously set target error.
             *
             * \return The target error
             */
            qreal targetError() const;


            /*!
             * \brief Sets the target mean squared error
             *
             * \param[in] targetError The target MSE
             *
             * \return `*this`
             */
            TrainingSet &targetError(const qreal &targetError);


            /*!
             * \brief Returns the maximum number of epochs that are allowed
             *  for training.
             *
             * \return The maximum number of iterations the training algorithm
             *  may use
             */
            size_t maxEpochs() const;


            /*!
             * \brief Sets the maximum number of iterations a training
             *  algorithm may use
             *
             * \param[in] maxEpochs The maximum number of epochs
             *
             * \return `*this`
             */
            TrainingSet &maxEpochs(const size_t &maxEpochs);


            /*!
             * Returns the number of epochs needed to complete
             * the training.
             */
            size_t epochs() const;


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
             * \brief Clears the TrainingSet completeley
             *  and resets all parameters
             */
            virtual void clear() override;


            /*!
             * \brief Deserializes an TrainingSet into the current object
             *
             * \param[in] json The JSON representation of the TrainingSet
             */
            virtual void fromJSON(const QJsonDocument &json) override;


            /*!
             * \brief Serializes the current TrainingSet
             *
             * \return The serialized version of the object
             */
            virtual QJsonDocument toJSON() const override;


        private:


            //! The target MSE
            qreal m_targetError;


            //! Maximum number of epochs the training will run for.
            size_t m_maxNumEpochs;


            //! Actual number of epochs it took to complete the training.
            size_t m_epochs;


            //! The actual MSE after training ran
            qreal m_error;
        };
    } /* namespace ANN */


    template <>
    struct JsonSchema<Winzent::ANN::TrainingSet>
    {
        static constexpr const char schemaURI[] =
                ":/schema/TrainingSetSchema.json";
    };
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
