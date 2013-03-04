#ifndef MOCK_LINEARNEURALNETWORKPATTERN_H
#define MOCK_LINEARNEURALNETWORKPATTERN_H


#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


namespace Mock {
    

    /*!
     * A mock neural network pattern that consists only of linear activation
     * functions and has weights of 1.0.
     */
    class LinearNeuralNetworkPattern : public Winzent::ANN::NeuralNetworkPattern
    {
    public:
        LinearNeuralNetworkPattern();


        /*!
         * Fully connects each layer to the next one: input => hidden => output,
         * without shortcut paths or recurrency.
         */
        virtual void configureNetwork(Winzent::ANN::NeuralNetwork *);
    };
    
} // namespace Mock

#endif // MOCK_LINEARNEURALNETWORKPATTERN_H
