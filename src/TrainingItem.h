#ifndef TRAININGITEM_H
#define TRAININGITEM_H


#include <ostream>

#include "Vector.h"
#include "LibVariantSupport.h"


namespace Winzent {
    namespace ANN {

        /*!
         * \brief Represents one training input and expected output
         *
         * This class represents one item a net can train on. It
         * contains the input values and the expected output values.
         * It also contains a method to calculate the RMSE if an
         * actual output is given.
         */
        class TrainingItem
        {
            friend libvariant::Variant to_variant<>(TrainingItem const&);


        public:


            /*!
             * \brief Constructs a new instance given input and expected
             *  output.
             */
            TrainingItem(Vector const& input, Vector const& expectedOutput);


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
            TrainingItem(Vector const& input);


            /*!
             * \brief Constructs a new, empty training item.
             */
            explicit TrainingItem();


            /*!
             * \brief Creates a copy of the other training item.
             */
            TrainingItem(TrainingItem const& rhs);


            /*!
             * \brief Move constructor
             */
            TrainingItem(TrainingItem&& rhs);


            /*!
             * \return A copy of the input for the Neural Network
             */
            Vector input() const;


            /*!
             * \return A copy of the output that is expected of the Neural
             *  Network
             */
            Vector expectedOutput() const;


            /*!
             * \return Whether an expected output exists, i.e., whether the
             *  output of the neural network is relevant or not.
             */
            bool outputRelevant() const;


            //! \brief Assignment operator
            TrainingItem& operator =(TrainingItem const& rhs);


        private:


            //! \brief The input presented to the network
            Vector m_input;


            //! \brief The output expected from the network.
            Vector m_expectedOutput;
        };


        template <> inline libvariant::Variant
        to_variant(TrainingItem const& trainingItem)
        {
            libvariant::Variant variant;

            variant["input"] = to_variant(trainingItem.m_input);
            variant["expectedOutput"] = to_variant(
                    trainingItem.m_expectedOutput);
        }


        template <> inline TrainingItem
        from_variant(libvariant::Variant const& variant)
        {
            return TrainingItem(
                    from_variant<Vector>(variant["input"]),
                    variant.Contains("expectedOutput")
                        ? from_variant<Vector>(variant["expectedOutput"])
                        : Vector());
        }
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream& operator <<(
            ostream& os,
            Winzent::ANN::TrainingItem const& trainingItem);
}

#endif // TRAININGITEM_H
