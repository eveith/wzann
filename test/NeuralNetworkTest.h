#ifndef NEURALNETWORKTEST_H_
#define NEURALNETWORKTEST_H_


#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


using Winzent::ANN::Vector;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::NeuralNetworkPattern;


namespace Mock {
    class NeuralNetworkTestDummyPattern: public NeuralNetworkPattern
    {
    public:

        static const int numLayers;


        NeuralNetworkTestDummyPattern();
        virtual ~NeuralNetworkTestDummyPattern() {}


        int numNeuronsInLayer(const int &layer)
        {
            return layer+1;
        }


        virtual void configureNetwork(NeuralNetwork &network) override;


        virtual Vector calculate(
                Winzent::ANN::NeuralNetwork &network,
                const Vector& input)
                override;

        virtual NeuralNetworkPattern* clone() const;
    };
}


#endif /* NEURALNETWORKTEST_H_ */
