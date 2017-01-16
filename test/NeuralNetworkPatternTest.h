#ifndef NEURALNETWORKPATTERNTEST_H_
#define NEURALNETWORKPATTERNTEST_H_


#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


using Winzent::ANN::Vector;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::NeuralNetworkPattern;


namespace Mock {
    class NeuralNetworkPatternTestDummyPattern: public NeuralNetworkPattern
    {
    public:

        int numLayers;
        int numNeuronsPerLayer;


        NeuralNetworkPatternTestDummyPattern();


        virtual Vector calculate(
                NeuralNetwork &network,
                const Vector &input)
                override;


        virtual NeuralNetworkPattern* clone() const;


        virtual void configureNetwork(NeuralNetwork &network) override;
    };
}


#endif
