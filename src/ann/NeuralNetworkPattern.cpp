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
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "Layer.h"
#include "Exception.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {
        NeuralNetworkPattern::NeuralNetworkPattern()
        {
        }


        NeuralNetworkPattern::NeuralNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction *> activationFunctions):
                    QObject(),
                    JsonSerializable(),
                    m_layerSizes(layerSizes),
                    m_activationFunctions(activationFunctions)
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


        void NeuralNetworkPattern::clear()
        {
            m_layerSizes.clear();
            for (auto &f: m_activationFunctions) {
                delete f;
            }
        }


        QJsonDocument NeuralNetworkPattern::toJSON() const
        {
            QJsonObject o;

            o["type"] = metaObject()->className();

            QJsonArray layerSizes;
            for (const auto &i: m_layerSizes) {
                layerSizes.push_back(i);
            }
            o["layerSizes"] = layerSizes;

            QJsonArray activationFunctions;
            for (const auto &i: m_activationFunctions) {
                activationFunctions.push_back(i->toJSON().object());
            }
            o["activationFunctions"] = activationFunctions;

            return QJsonDocument(o);
        }


        void NeuralNetworkPattern::fromJSON(const QJsonDocument &json)
        {

        }
    }
}
