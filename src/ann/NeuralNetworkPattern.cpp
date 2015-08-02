/*!
 * \file	NeuralNetworkPattern.cpp
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */


#include <initializer_list>

#include <QObject>
#include <QList>
#include <QString>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Connection.h"
#include "Exception.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {


        qreal NeuralNetworkPattern::weightRandomMin = -0.5;
        qreal NeuralNetworkPattern::weightRandomMax =  0.5;


        NeuralNetworkPattern::NeuralNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction *> activationFunctions):
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
                initializer_list<ActivationFunction *> activationFunctions):
                    NeuralNetworkPattern(
                        QList<int>(layerSizes),
                        QList<ActivationFunction *>(activationFunctions))
        {
        }


        NeuralNetworkPattern::~NeuralNetworkPattern()
        {
            for (auto &f: m_activationFunctions) {
                delete f;
            }
        }


        void NeuralNetworkPattern::fullyConnectNetworkLayers(
                NeuralNetwork *network,
                const int &fromLayer,
                const int &toLayer)
        {
            int fromLayerSize   = m_layerSizes.at(fromLayer);
            int toLayerSize     = m_layerSizes.at(toLayer);

            // Iterate over all neurons:

            for (int i = 0; i != fromLayerSize; ++i) {
                for (int j = 0; j != toLayerSize; ++j) {
                    network->connectNeurons(
                            network->layerAt(fromLayer)->neuronAt(i),
                            network->layerAt(toLayer)->neuronAt(j))
                                    ->weight(0.0);
                }
            }
        }


        QString NeuralNetworkPattern::toString()
        {
            return "unknown: TODO FIXME";
        }
    }
}
