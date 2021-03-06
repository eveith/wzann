#ifndef WZANN_WEIGHTRANDOMIZER_H_
#define WZANN_WEIGHTRANDOMIZER_H_


namespace wzann {
    class NeuralNetwork;


    /*!
     * \brief Interface for ANN weight randomizer strategies
     *
     * This interface provides a common method for weight randomization
     * strategies that initializes the weights of an artificial neural
     * network.
     */
    class WeightRandomizer
    {
    public:


        /*!
         * \brief Randomizes the weights of the given ANN according to the
         *  implementing strategy.
         *
         * \param[in] neuralNetwork The neural network
         */
        virtual void randomize(NeuralNetwork& neuralNetwork) = 0;
    };
} // namespace wzann


#endif // WZANN_WEIGHTRANDOMIZER_H_
