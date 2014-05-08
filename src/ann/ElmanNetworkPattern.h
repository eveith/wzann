/*!
 * \file	ElmanNetworkPattern.h
 * \brief
 * \date	31.12.2012
 * \author	eveith
 */

#ifndef ELMANNETWORKPATTERN_H_
#define ELMANNETWORKPATTERN_H_


#include <initializer_list>
#include <QList>

#include "NeuralNetworkPattern.h"


using std::initializer_list;


namespace Winzent
{
    namespace ANN
    {

        /*!
         * Represents the general layout and wokring mode of an
         * <em>Elman</em> network.
         *
         * Elman networks have three layers,
         * the input, hidden and output layer. Each neuron in the
         * hidden layer is accompanied by a neuron in the so-called
         * <em>context layer</em>. Context neurons remember the last
         * output of its corresponding hidden neuron and feed it
         * again to the hidden neuron upon next activation.
         */
        class ElmanNetworkPattern: public NeuralNetworkPattern
        {
            Q_OBJECT

        public:


            /*!
             * Index constants for the layers we generate.
             */
            enum Layers {
                INPUT,
                CONTEXT,
                HIDDEN,
                OUTPUT
            };


        protected:


            /*!
             * Feed-forward calculation of an Elman network.
             */
            virtual ValueVector calculate(
                    NeuralNetwork *const &network,
                    const ValueVector &input);


        public:


            /*!
             * Creates a new network pattern.
             *
             * This constructor automagically takes care of the
             * context layer, i.e. you do not have to supply
             * the definition for this one manually.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            ElmanNetworkPattern(
                    QList<int> layerSizes,
                    QList<ActivationFunction*> activationFunctions,
                    QObject *parent = 0);


            /*!
             * Creates a new network pattern.
             *
             * This constructor automagically takes care of the
             * context layer, i.e. you do not have to supply
             * the definition for this one manually.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            ElmanNetworkPattern(
                    initializer_list<int> layerSizes,
                    initializer_list<ActivationFunction*> activationFunctions,
                    QObject *parent = 0);


            /*!
             * Creates a clone of this pattern, cast to the base
             * class.
             */
            virtual NeuralNetworkPattern* clone() const;


            /*!
             * Configures the neural network to be an Elman network.
             *
             * It introduces four layers (input, context, hidden and
             * output layer) and sets the connections up:
             *
             * - Each neuron of the input layer is fully connected to
             *   each hidden layer neuron.
             * - Each neuron of the context layer is fully connected
             *   to each neuron of the hidden layer.
             * - Each neuron of the hidden layer is connected to
             *   exactly one neuron of the context layer (1:1).
             * - Each hidden layer neuron is connected to each
             *   output layer neuron.
             */
            virtual void configureNetwork(NeuralNetwork *network);
        };

    } /* namespace ANN */
} /* namespace Winzent */
#endif /* ELMANNETWORKPATTERN_H_ */
