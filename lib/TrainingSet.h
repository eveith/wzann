#ifndef WZANN_TRAININGSET_H_
#define WZANN_TRAININGSET_H_


#include <cstddef>
#include <ostream>

#include "WzannGlobal.h"
#include "TrainingItem.h"
#include "NeuralNetwork.h"
#include "JsonSerializable.h"
#include "LibVariantSupport.h"


namespace wzann {
    class TrainingAlgorithm;


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
        friend TrainingSet from_variant<>(libvariant::Variant const&);
        friend TrainingSet* new_from_variant<>(libvariant::Variant const&);


    public:


        //! \brief A vector of training items
        typedef std::vector<TrainingItem> TrainingItems;


        //! \brief The actual set of training data
        TrainingItems trainingItems;


        /*!
         * Constructs a new TrainingSet by supplying the training
         * data and the relevant paramters.
         *
         * \param[in] trainingData The data used for training, given
         *  as a Hash mapping input to output. Both are
         *  <code>double[]</code>, where each array index
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
                TrainingItems trainingItems,
                double targetError,
                size_t maxNumEpochs);


        //! Creates an empty training set.
        explicit TrainingSet();


        //! \brief Copy constructor
        TrainingSet(const TrainingSet &other);


        /*!
         * \brief Returns the mean square error after the current
         *  training epoch.
         *
         * \return The actual, current error
         */
        double error() const;


        /*!
         * \brief Returns the previously set target error.
         *
         * \return The target error
         */
        double targetError() const;


        /*!
         * \brief Sets the target mean squared error
         *
         * \param[in] targetError The target MSE
         *
         * \return `*this`
         */
        TrainingSet &targetError(double targetError);


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
        TrainingSet& maxEpochs(size_t maxEpochs);


        /*!
         * Returns the number of epochs needed to complete
         * the training.
         */
        size_t epochs() const;


        /*!
         * \brief Move-adds a training item to this training set
         *
         * \param[in] item The training item
         */
        void push_back(TrainingItem&& item);


        /*!
         * \brief Copies all training items from the other training set
         *  to this one, in the correct order.
         *
         * \param[in] trainingSet The training set whose data should be
         *  appended to this one.
         */
        void push_back(TrainingSet const& trainingSet);


        /*!
         * \brief Adds the given TrainingItem to the list of
         *  training items by moving it into the TrainingSet.
         *
         * \param[in] item The new item
         *
         * \return `*this`
         */
        TrainingSet& operator <<(TrainingItem&& item);


        /*!
         * \brief Assignes another TrainingSet to this one
         *
         * \param[in] rhs The other training set to copy from
         *
         * \return `*this`
         */
        TrainingSet& operator =(const TrainingSet &rhs);


    private:


        //! The target MSE
        double m_targetError;


        //! Maximum number of epochs the training will run for.
        size_t m_maxNumEpochs;


        //! Actual number of epochs it took to complete the training.
        size_t m_epochs;


        //! The actual MSE after training ran
        double m_error;
    };


    template <>
    inline libvariant::Variant to_variant(TrainingSet const& ts)
    {
        libvariant::Variant v;

        v["epochs"] = ts.epochs();
        v["targetError"] = ts.targetError();
        v["maxEpochs"] = ts.maxEpochs();
        v["error"] = ts.error();

        libvariant::Variant::List trainingItems;
        for (auto const& i: ts.trainingItems) {
            trainingItems.push_back(to_variant(i));
        }
        v["trainingItems"] = trainingItems;

        return v;
    }


    template <>
    inline TrainingSet from_variant(libvariant::Variant const& variant)
    {
        TrainingSet ts;

        ts.m_epochs = variant.Contains("epochs")
                ? static_cast<size_t>(variant["epochs"].AsUnsigned())
                : 0u;
        ts.maxEpochs(static_cast<size_t>(
                variant["maxEpochs"].AsUnsigned()));

        ts.m_error = variant.Contains("error")
                ? variant["error"].AsDouble()
                : std::numeric_limits<double>::max();
        ts.targetError(variant["targetError"].AsDouble());

        for (const auto &i: variant["trainingItems"].AsList()) {
            ts.push_back(from_variant<TrainingItem>(i));
        }

        return ts;
    }


    template <>
    inline TrainingSet* new_from_variant(libvariant::Variant const& variant)
    {
        TrainingSet* ts = new TrainingSet();

        ts->m_epochs = variant.Contains("epochs")
                ? static_cast<size_t>(variant["epochs"].AsUnsigned())
                : 0u;
        ts->maxEpochs(static_cast<size_t>(
                variant["maxEpochs"].AsUnsigned()));
        ts->m_error = variant.Contains("error")
                ? variant["error"].AsDouble()
                : std::numeric_limits<double>::max();
        ts->targetError(variant["targetError"].AsDouble());

        for (const auto& i : variant["trainingItems"].AsList()) {
            ts->push_back(from_variant<TrainingItem>(i));
        }

        return ts;
    }


    template <>
    struct JsonSchema<wzann::TrainingSet>
    {
        static constexpr const char schemaURI[] = WZANN_SCHEMA_PATH
                "/TrainingSetSchema.json";
    };
} // namespace wzann


namespace std {
    ostream& operator <<(
            ostream& os,
            wzann::TrainingSet::TrainingItems const& trainingData);
    ostream& operator <<(
            ostream& os,
            wzann::TrainingSet const& trainingSet);
}

#endif /* WZANN_TRAININGSET_H_ */
