#ifndef NEURALNETWORKPATTERNTEST_H_
#define NEURALNETWORKPATTERNTEST_H_


#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


using Winzent::ANN::Vector;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::NeuralNetworkPattern;


namespace Mock {
    class NeuralNetworkPatternTestDummyPattern : public NeuralNetworkPattern
    {
    public:

        size_t numLayers;
        size_t numNeuronsPerLayer;


        NeuralNetworkPatternTestDummyPattern();


        virtual Vector calculate(
                NeuralNetwork& network,
                Vector const& input)
                override;


        virtual NeuralNetworkPattern* clone() const override;


        virtual void configureNetwork(NeuralNetwork &network) override;
    };
}


#endif
