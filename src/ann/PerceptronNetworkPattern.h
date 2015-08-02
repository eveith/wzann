/*!
 * \file
 */


#ifndef WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
#define WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H


#include <initializer_list>

#include <QList>

#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {

        class ActivationFunction;

        /*!
         * Instances of this class represent the pattern to create a
         * (potentially multi-layered), feed-forward perceptron without
         * recurrent connections, shortcut connections or other specialities.
         */
        class PerceptronNetworkPattern: public NeuralNetworkPattern
        {
        protected:


            /*!
             * \brief Configures the supplied neural network
             *  to be an perceptron.
             */
            virtual void configureNetwork(NeuralNetwork *network) override;


            /*!
             * \sa NeuralNetworkPattern#calculate
             */
            virtual Vector calculate(
                    NeuralNetwork *const &network,
                    const Vector &input)
                    override;


        public:


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    QList<int> layerSizes,
                    QList<ActivationFunction *> activationFunctions);


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    initializer_list<int> layerSizes,
                    initializer_list<ActivationFunction *>
                        activationFunctions);


            /*!
             * \brief Creates a clone of a pattern instance.
             *
             * \sa NeuralNetworkPattern#clone
             */
            virtual NeuralNetworkPattern *clone() const override;
        };

    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
