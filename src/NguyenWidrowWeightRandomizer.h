/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de<
 * \date 2013-03-28
 */


#ifndef NGUYENWIDROWWEIGHTRANDOMIZER_H
#define NGUYENWIDROWWEIGHTRANDOMIZER_H


#include "WeightRandomizer.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class Layer;


        /*!
         * Randomizes the weights of a neural network based on the algorithm
         * developed by Nguyen and Widrow.
         */
        class NguyenWidrowWeightRandomizer: public WeightRandomizer
        {
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
                    NeuralNetwork &network,
                    Layer &from,
                    Layer &to)
                    const;


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
            void randomize(NeuralNetwork &network);
        };
    }
}

#endif // NGUYENWIDROWWEIGHTRANDOMIZER_H
