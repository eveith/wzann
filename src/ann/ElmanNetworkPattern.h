#ifndef WINZENT_ANN_ELMANNETWORKPATTERN_H_
#define WINZENT_ANN_ELMANNETWORKPATTERN_H_


#include <initializer_list>

#include <QObject>
#include <QList>

#include <ClassRegistry.h>

#include "NeuralNetworkPattern.h"

#include "Winzent-ANN_global.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {

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
        class WINZENTANNSHARED_EXPORT ElmanNetworkPattern:
                public NeuralNetworkPattern
        {
            Q_OBJECT

            friend class Winzent::ClassRegistration<ElmanNetworkPattern>;


        public:


            //! \brief Index constants for the layers we generate.
            enum Layers {
                INPUT,
                CONTEXT,
                HIDDEN,
                OUTPUT
            };


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
                    QList<ActivationFunction *> activationFunctions);


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
                    initializer_list<ActivationFunction *>
                        activationFunctions);


            //! \brief Clones this pattern
            virtual NeuralNetworkPattern* clone() const override;


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
            virtual void configureNetwork(NeuralNetwork *const &network)
                    override;


        protected:


            /*!
             * Feed-forward calculation of an Elman network.
             */
            virtual Vector calculate(
                    NeuralNetwork *const &network,
                    const Vector &input)
                    override;


        private:


            ElmanNetworkPattern();
        };
    } /* namespace ANN */
} /* namespace Winzent */


#endif /* WINZENT_ANN_ELMANNETWORKPATTERN_H_ */
