/*!
 * \file	NeuralNetworkTest.h
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#ifndef NEURALNETWORKTEST_H_
#define NEURALNETWORKTEST_H_


#include <qobject.h>

#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"


using Winzent::ANN::Vector;
using Winzent::ANN::NeuralNetwork;
using Winzent::ANN::NeuralNetworkPattern;


namespace Mock {
    class NeuralNetworkTestDummyPattern: public NeuralNetworkPattern
    {
        Q_OBJECT

    public:

        static const int numLayers;


        NeuralNetworkTestDummyPattern();
        virtual ~NeuralNetworkTestDummyPattern() {}


        int numNeuronsInLayer(const int &layer) {
            return layer+1;
        }


        virtual void configureNetwork(NeuralNetwork* const& network);


        virtual Vector calculate(
                NeuralNetwork *const &network,
                const Vector& input);

        virtual NeuralNetworkPattern* clone() const;
    };
}


class NeuralNetworkTest: public QObject
{
    Q_OBJECT


private slots:

    void testLayerAdditionRemoval();
    void testCalculateLayer();
    void testCalculateLayerTransition();
    void testSerialization();
    void testInitialLayerSize();
    void testConnectionsFromTo();
    void testClone();
    void testEachLayerIterator();
    void testEachConnectionIterator();
    void testOperatorEquals();
    void testJsonSerialization();
};

#endif /* NEURALNETWORKTEST_H_ */
