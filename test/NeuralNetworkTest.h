#ifndef NEURALNETWORKTEST_H_
#define NEURALNETWORKTEST_H_


#include "ClassRegistry.h"

#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


using wzann::Vector;
using wzann::NeuralNetwork;
using wzann::NeuralNetworkPattern;


namespace Mock {
    class NeuralNetworkTestDummyPattern : public NeuralNetworkPattern
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
                wzann::NeuralNetwork &network,
                const Vector& input)
                override;

        virtual NeuralNetworkPattern* clone() const override;
    };
}


WZANN_REGISTER_CLASS(
        Mock::NeuralNetworkTestDummyPattern,
        wzann::NeuralNetworkPattern)


#endif /* NEURALNETWORKTEST_H_ */
