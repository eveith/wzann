#ifndef WINZENT_ANN_SIMPLEWEIGHTRANDOMIZER_H
#define WINZENT_ANN_SIMPLEWEIGHTRANDOMIZER_H


#include "WeightRandomizer.h"


namespace wzann {
    class NeuralNetwork;


    class SimpleWeightRandomizer : public WeightRandomizer
    {
    public:


        /*!
         * \brief The default minimum weight
         */
        static constexpr double defaultMinWeight = -0.1;


        /*!
         * \brief The default maximum weight
         */
        static constexpr double defaultMaxWeight = +0.1;


        /*!
         * \brief Constructs a new instance of the simple weight
         *  randomizer
         *
         * \sa #minWeight()
         *
         * \sa #maxWeight()
         */
        SimpleWeightRandomizer();


        /*!
         * \brief Retrieves the absolute minimum weight.
         *
         * \return The smallest possible weight value
         */
        double minWeight() const;


        /*!
         * \brief Sets the absolute minimum weight.
         *
         * \param[in] weight The new weight
         *
         * \return `*this`
         */
        SimpleWeightRandomizer& minWeight(double weight);


        /*!
         * \brief Retrieves the absolute maximum weight.
         *
         * \return The smallest possible weight value
         */
        double maxWeight() const;


        /*!
         * \brief Sets the absolute maximum weight.
         *
         * \param[in] weight The new weight
         *
         * \return `*this`
         */
        SimpleWeightRandomizer& maxWeight(double weight);


        /*!
         * \brief Randomizes the weights of the artificial neural network
         *  by assigning a random number from [minWeight, maxWeight) to
         *  each non-weightfixed connection
         *
         * \param[in] neuralNetwork The neural network
         *
         * \sa #minWeight()
         *
         * \sa #maxWeight()
         */
        virtual void randomize(NeuralNetwork& neuralNetwork) override;


    private:


        //! \brief The absolute minimum weight of this instance
        double m_minWeight;


        //! \brief The absolute maximum weight of this instance
        double m_maxWeight;
    };
} // namespace wzann

#endif // WINZENT_ANN_SIMPLEWEIGHTRANDOMIZER_H
