/*!
 * \file	NeuralNetworkPatternTest.h
 * \brief
 * \date	03.01.2013
 * \author	eveith
 */

#ifndef NEURALNETWORKPATTERNTEST_H_
#define NEURALNETWORKPATTERNTEST_H_

#include <qobject.h>
#include <QVector>

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
                NeuralNetwork *const &network,
                const Vector& input);


        virtual NeuralNetworkPattern* clone() const;


        virtual void configureNetwork(NeuralNetwork *network);
    };
}


class NeuralNetworkPatternTest: public QObject
{
    Q_OBJECT

public:


private slots:

    void testFullyConnectNetworkLayers();
};


#endif
