#include "Layer.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"


namespace Winzent {
    namespace ANN {
        NeuralNetworkPattern::NeuralNetworkPattern()
        {
        }


        NeuralNetworkPattern& NeuralNetworkPattern::addLayer(
                NeuralNetworkPattern::SimpleLayerDefinition layerDefinition)
        {
            m_layerDefinitions.push_back(layerDefinition);
            return *this;
        }


        void NeuralNetworkPattern::fullyConnectNetworkLayers(
                Layer& from,
                Layer& to)
        {
            assert(from.parent() == to.parent());
            auto* neuralNetwork = from.parent();

            for (auto const& fromNeuron: from) {
                for (auto const& toNeuron: to) {
                    neuralNetwork->connectNeurons(fromNeuron, toNeuron)
                            .weight(0.0);
                }
            }
        }


        bool NeuralNetworkPattern::operator ==(
                NeuralNetworkPattern const& other)
                const
        {
            return m_layerDefinitions == other.m_layerDefinitions;
        }


        bool NeuralNetworkPattern::operator !=(
                NeuralNetworkPattern const& other)
                const
        {
            return !(*this == other);
        }
    }
}
