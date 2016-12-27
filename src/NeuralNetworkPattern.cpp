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

#include <ClassRegistry.h>

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
                NeuralNetworkPattern::LayerSizes const& layerSizes,
                NeuralNetworkPattern::ActivationFunctions const&
                        activationFunctions):
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


        NeuralNetworkPattern& NeuralNetworkPattern::add(
                NeuralNetworkPattern::LayerDefinition const& layerDefinition)
        {
            m_layerSizes.push_back(layerDefinition.first);
            m_activationFunctions.push_back(layerDefinition.second.clone());

            return *this;
        }


        void NeuralNetworkPattern::fullyConnectNetworkLayers(
                NeuralNetwork &network,
                const int &fromLayer,
                const int &toLayer)
        {
            int fromLayerSize   = m_layerSizes.at(fromLayer);
            int toLayerSize     = m_layerSizes.at(toLayer);

            // Iterate over all neurons:

            for (int i = 0; i != fromLayerSize; ++i) {
                for (int j = 0; j != toLayerSize; ++j) {
                    network.connectNeurons(
                            network[fromLayer][i],
                            network[toLayer][j])
                        .weight(0);
                }
            }
        }


        void NeuralNetworkPattern::clear()
        {
            m_layerSizes.clear();
            for (auto &f: m_activationFunctions) {
                delete f;
            }
            m_activationFunctions.clear();
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
            clear();
            QJsonObject o = json.object();

            QJsonArray layerSizes = o["layerSizes"].toArray();
            for (const auto& i: layerSizes) {
                m_layerSizes.push_back(i.toInt());
            }

            auto classRegistry =
                    ClassRegistry<ActivationFunction>::instance();
            QJsonArray activationFunctions =
                    o["activationFunctions"].toArray();
            for (const auto& i: activationFunctions) {
                auto activationFunction = classRegistry->create(
                        i.toObject()["type"].toString());
                Q_ASSERT(nullptr != activationFunction);

                activationFunction->fromJSON(QJsonDocument(i.toObject()));
                m_activationFunctions.push_back(activationFunction);
            }
        }


        bool NeuralNetworkPattern::equals(
                const NeuralNetworkPattern* const& other)
                const
        {
            bool equal = true;

            equal &= (m_layerSizes == other->m_layerSizes);
            equal &= (m_activationFunctions.size()
                    == other->m_activationFunctions.size());

            if (! equal) {
                return equal;
            }

            for (auto i1 = m_activationFunctions.constBegin(),
                        i2 = other->m_activationFunctions.constBegin();
                    i1 != m_activationFunctions.constEnd()
                        && i2 != other->m_activationFunctions.constEnd();
                    i1++, i2++) {
                if (! (*i1)->equals(*i2)) {
                    return false;
                }
            }

            return equal;
        }
    }
}
