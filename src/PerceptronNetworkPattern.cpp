#include <utility>

#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ClassRegistry.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"
#include "PerceptronNetworkPattern.h"


using std::get;


namespace Winzent {
    namespace ANN {
        PerceptronNetworkPattern::PerceptronNetworkPattern()
        {
        }


        PerceptronNetworkPattern::~PerceptronNetworkPattern()
        {
        }


        NeuralNetworkPattern* PerceptronNetworkPattern::clone() const
        {
            auto* patternClone = new PerceptronNetworkPattern();

            for (auto const& layerDefinition: m_layerDefinitions) {
                patternClone->addLayer(
                        SimpleLayerDefinition(layerDefinition));
            }

            return patternClone;
        }


        bool PerceptronNetworkPattern::operator ==(
                NeuralNetworkPattern const& other)
                const
        {
            return reinterpret_cast<PerceptronNetworkPattern const*>(&other)
                        != nullptr
                    && NeuralNetworkPattern::operator ==(other);
        }


        void PerceptronNetworkPattern::configureNetwork(
                NeuralNetwork& network)
        {
            // Add the layers & neurons:

            for (auto const& layerDefinition: m_layerDefinitions) {
                auto* layer = new Layer();

                for (Layer::size_type i = 0; i != get<0>(layerDefinition);
                        ++i) {
                    auto* neuron = new Neuron();
                    neuron->activationFunction(get<1>(layerDefinition));
                    layer->addNeuron(neuron);
                }

                network << layer;
            }

            // Now connect layers:

            for (NeuralNetwork::size_type i = 0; i != network.size(); ++i) {
                if (i > 0) {
                    for (auto const& neuron: network[i]) {
                        network.connectNeurons(
                                network.biasNeuron(),
                                neuron)
                            .weight(-1.0)
                            .fixedWeight(false);
                    }
                }

                if (i + 1 < network.size()) {
                    fullyConnectNetworkLayers(network[i], network[i+1]);
                }
            }
        }


        Vector PerceptronNetworkPattern::calculate(
                NeuralNetwork& network,
                Vector const& input)
        {
            Vector output = input; // For the loop

            for (size_t i = 0; i != network.size(); ++i) {
                output = network.calculateLayer(
                        network[i],
                        output);

                if (i < network.size() - 1) {
                    output = network.calculateLayerTransition(
                            network[i],
                            network[i+1],
                            output);
                }
            }

            return output;
        }
    } // namespace ANN
} // namespace Winzent


WINZENT_REGISTER_CLASS(
        Winzent::ANN::PerceptronNetworkPattern,
        Winzent::ANN::NeuralNetworkPattern)
