/*!
 * \file	NeuralNetworkPattern.cpp
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */


#include <initializer_list>

#include <QDebug>
#include <QObject>
#include <QList>
#include <QString>

#include "Exception.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"


using std::initializer_list;


namespace Winzent
{
    namespace ANN
    {


        double NeuralNetworkPattern::weightRandomMin = -0.5;
        double NeuralNetworkPattern::weightRandomMax = +0.5;


        NeuralNetworkPattern::NeuralNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction*> activationFunctions,
                QObject *parent):
                        QObject(parent),
                        m_layerSizes(layerSizes),
                        m_activationFunctions(activationFunctions),
                        m_weightRandomMin(weightRandomMin),
                        m_weightRandomMax(weightRandomMax)
        {
            if (m_layerSizes.size() != m_activationFunctions.size()) {
                throw LayerSizeMismatchException(
                        m_layerSizes.size(),
                        m_activationFunctions.size());
            }
        }


        NeuralNetworkPattern::NeuralNetworkPattern(
                initializer_list<int> layerSizes,
                initializer_list<ActivationFunction*> activationFunctions,
                QObject *parent):
                        NeuralNetworkPattern(
                                QList<int>(layerSizes),
                                QList<ActivationFunction*>(activationFunctions),
                                parent)
        {
        }


        NeuralNetworkPattern::~NeuralNetworkPattern()
        {
        }


        void NeuralNetworkPattern::fullyConnectNetworkLayers(
                NeuralNetwork *network,
                const int &fromLayer,
                const int &toLayer)
        {
            int fromLayerSize = m_layerSizes.at(fromLayer);
            int toLayerSize = m_layerSizes.at(toLayer);

            // Iterate over all neurons. Begin at index 1, because index 0
            // is always the bias neuron.

            for (int i = 1; i != fromLayerSize; ++i) {
                    for (int j = 1; j != toLayerSize; ++j) {
                        network->connectNeurons(
                                network->translateIndex(fromLayer, i),
                                network->translateIndex(toLayer, j));
                    }
            }
        }


        QString NeuralNetworkPattern::toString()
        {
            return QString(metaObject()->className());
        }
    }
}
