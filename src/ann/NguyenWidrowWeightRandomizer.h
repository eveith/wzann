/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de<
 * \date 2013-03-28
 */


#ifndef NGUYENWIDROWWEIGHTRANDOMIZER_H
#define NGUYENWIDROWWEIGHTRANDOMIZER_H


#include <QObject>


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class Layer;


        /*!
         * Randomizes the weights of a neural network based on the algorithm
         * developed by Nguyen and Widrow.
         */
        class NguyenWidrowWeightRandomizer: public QObject
        {
            Q_OBJECT

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
                    NeuralNetwork *const &network,
                    Layer *const &from,
                    Layer *const &to)
                    const;


        public:


            /*!
             * Constructs a new randomizer. Typically, you'll just want to
             * call `NguyenWidrowWeightRandomizer().randomize(network);`.
             */
            explicit NguyenWidrowWeightRandomizer(QObject *parent = 0);


            /*!
             * Randomizes the network's weights according to the Nguyen-Widrow
             * algorithm.
             *
             * \param network The network whose weights shall be randomized.
             */
            void randomize(NeuralNetwork *const &network) const;
        };
    }
}

#endif // NGUYENWIDROWWEIGHTRANDOMIZER_H
