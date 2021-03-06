#ifndef WZANN_NGUYENWIDROWWEIGHTRANDOMIZER_H_
#define WZANN_NGUYENWIDROWWEIGHTRANDOMIZER_H_


#include "WeightRandomizer.h"


namespace wzann {
    class NeuralNetwork;
    class Layer;


    /*!
     * \brief Randomizes the weights of a neural network based on
     *  the algorithm developed by Nguyen and Widrow.
     */
    class NguyenWidrowWeightRandomizer : public WeightRandomizer
    {
    public:


        /*!
         * \brief Constructs a new randomizer.
         *
         * Typically, you'll just want to call
         * `NguyenWidrowWeightRandomizer().randomize(network);`.
         */
        NguyenWidrowWeightRandomizer();


        /*!
         * \brief Randomizes the network's weights according to the
         *  Nguyen-Widrow algorithm.
         *
         * \param[inout] network The network whose weights
         *  shall be randomized.
         */
        void randomize(NeuralNetwork& network);


    private:


        /*!
         * Randomizes one synapse, i.e. the connection from one layer to
         * another.
         *
         * \param[in] network The neural network that contains both layers
         *
         * \param[in] from The layer from which the connections originate
         *
         * \param[in] to The layer where all connections lead to.
         */
        void randomizeSynapse(
                NeuralNetwork& network,
                Layer& from,
                Layer& to)
                const;

    };
} // namespace wzann

#endif // WZANN_NGUYENWIDROWWEIGHTRANDOMIZER_H_
