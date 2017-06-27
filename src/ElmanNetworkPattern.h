#ifndef WINZENT_ANN_ELMANNETWORKPATTERN_H_
#define WINZENT_ANN_ELMANNETWORKPATTERN_H_


#include "NeuralNetworkPattern.h"


namespace Winzent {
    namespace ANN {

        /*!
         * \brief This ANN pattern implements the layout of Elman Simple
         *  Recurrent Networks.
         *
         * Elman networks have three layers,
         * the input, hidden and output layer. Each neuron in the
         * hidden layer is accompanied by a neuron in the so-called
         * <em>context layer</em>. Context neurons remember the last
         * output of its corresponding hidden neuron and feed it
         * again to the hidden neuron upon next activation.
         */
        class ElmanNetworkPattern : public NeuralNetworkPattern
        {
        public:


            //! \brief Index constants for the layers we generate.
            enum Layers {
                INPUT = 0,
                CONTEXT,
                HIDDEN,
                OUTPUT
            };


            /*!
             * \brief Creates a new, empty pattern
             *
             * \sa NeuralNetworkPattern#addLayer()
             */
            explicit ElmanNetworkPattern();


            virtual ~ElmanNetworkPattern();


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
            virtual void configureNetwork(NeuralNetwork &network) override;


            /*!
             * \brief Checks for equality of two ElmanNetworkPattern
             *
             * \param[in] other The other pattern
             *
             * \return True if the two are of the same class and have the
             *  same parameters
             */
            virtual bool operator ==(NeuralNetworkPattern const& other)
                    const override;

        protected:


            /*!
             * \brief Feed-forward calculation of an Elman network.
             *
             * \sa NeuralNetworkPattern#calculate()
             */
            virtual Vector calculate(
                    NeuralNetwork& network,
                    Vector const& input)
                    override;
        };
    } /* namespace ANN */
} /* namespace Winzent */


#endif /* WINZENT_ANN_ELMANNETWORKPATTERN_H_ */
