/*!
 * \file
 */


#ifndef WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
#define WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H


#include <initializer_list>

#include <QObject>
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
        class PerceptronNetworkPattern : public NeuralNetworkPattern
        {
            Q_OBJECT

        protected:


            /*!
             * Configures the supplied neural network to be an perceptron.
             */
            virtual void configureNetwork(NeuralNetwork *network);


            /*!
             * \sa NeuralNetworkPattern#calculate
             */
            virtual ValueVector calculate(
                    NeuralNetwork *const &network,
                    const ValueVector &input);


        public:


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    QList<int> layerSizes,
                    QList<ActivationFunction*> activationFunctions,
                    QObject *parent = 0);


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    initializer_list<int> layerSizes,
                    initializer_list<ActivationFunction*> activationFunctions,
                    QObject *parent = 0);


            /*!
             * Creates a clone of a pattern instance.
             *
             * \sa NeuralNetworkPattern#clone
             */
            virtual NeuralNetworkPattern* clone() const;
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
