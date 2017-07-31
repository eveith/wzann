#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ClassRegistry.h"
#include "ActivationFunction.h"
#include "LayerSizeMismatchException.h"

#include "NeuralNetworkPattern.h"
#include "ElmanNetworkPattern.h"


namespace wzann {
    ElmanNetworkPattern::ElmanNetworkPattern() : NeuralNetworkPattern()
    {
    }


    ElmanNetworkPattern::~ElmanNetworkPattern()
    {
    }


    NeuralNetworkPattern* ElmanNetworkPattern::clone() const
    {
        auto* patternClone = new ElmanNetworkPattern();

        for (auto const& layerDefinition: m_layerDefinitions) {
            patternClone->addLayer(
                    SimpleLayerDefinition(layerDefinition));
        }

        return patternClone;
    }


    void ElmanNetworkPattern::configureNetwork(NeuralNetwork& network)
    {
        // Make sure that we do not get more than three layers here:

        if (m_layerDefinitions.size() != 3) {
            throw LayerSizeMismatchException(
                    3,
                    m_layerDefinitions.size());
        }

        // We need to insert the context layer after the input layer.

        m_layerDefinitions.insert(
                m_layerDefinitions.begin() + HIDDEN,
                m_layerDefinitions.at(1));

        // Create layers & neurons:

        for (auto const& layerDefinition: m_layerDefinitions) {
            auto* layer = new Layer();
            auto layerSize = layerDefinition.first;

            for (SimpleLayerDefinition::first_type i = 0; i != layerSize;
                    ++i) {
                auto* neuron = new Neuron();
                neuron->activationFunction(layerDefinition.second);
                layer->addNeuron(neuron);
            }

            network << layer;
        }

        // Set up connections:

        for (NeuralNetwork::size_type lidx = INPUT; lidx <= OUTPUT;
                ++lidx) {
            auto layerSize = m_layerDefinitions.at(lidx).first;

            switch (lidx) {
            case INPUT: {
                fullyConnectNetworkLayers(network[lidx], network[HIDDEN]);
                break;
            }
            case CONTEXT: {
                fullyConnectNetworkLayers(network[lidx], network[HIDDEN]);
                break;
            }
            case HIDDEN: {
                fullyConnectNetworkLayers(network[lidx], network[OUTPUT]);

                for (NeuralNetwork::size_type i = 0; i != layerSize;
                        ++i) {
                    auto &connection = network.connectNeurons(
                            network[HIDDEN][i],
                            network[CONTEXT][i]);
                    connection.weight(1.0);
                    connection.fixedWeight(true);
                }

                for (auto &neuron: network[lidx]) {
                    network.connectNeurons(
                            network.biasNeuron(),
                            neuron)
                        .weight(-1.0)
                        .fixedWeight(false);
                }

                break;
            }
            case OUTPUT: {
                for (auto &neuron: network[lidx]) {
                    network.connectNeurons(
                            network.biasNeuron(),
                            neuron)
                        .weight(-1.0).
                        fixedWeight(false);
                }

               break;
            }
            }
        }

        // Make sure that no connection between a hidden layer unit
        // and the BIAS neuron exists:

        for (Neuron const& neuron: network[CONTEXT]) {
            if (network.connectionExists(
                    network.biasNeuron(),
                    neuron)) {
                network.disconnectNeurons(network.biasNeuron(), neuron);
            }
        }
    }


    bool ElmanNetworkPattern::operator ==(
            NeuralNetworkPattern const& other)
            const
    {
        return reinterpret_cast<ElmanNetworkPattern const*>(&other) != nullptr
                && NeuralNetworkPattern::operator ==(other);
    }


    Vector ElmanNetworkPattern::calculate(
            NeuralNetwork& network,
            Vector const& input)
    {
        auto layerInput = network.calculateLayer(
                network[INPUT],
                input);
        layerInput = network.calculateLayerTransition(
                network[INPUT],
                network[HIDDEN],
                layerInput);

        // Fetch remembered values from the context layer:

        {
            auto &contextLayer = network[CONTEXT];
            Vector rememberedValues;
            rememberedValues.reserve(contextLayer.size());

            for (Layer::size_type i = 0; i != contextLayer.size(); ++i) {
                rememberedValues.push_back(contextLayer[i].lastInput());
            }

            rememberedValues = network.calculateLayerTransition(
                    network[CONTEXT],
                    network[HIDDEN],
                    rememberedValues);

            for (Vector::size_type i = 0; i != rememberedValues.size();
                    ++i) {
                layerInput[i] += rememberedValues[i];
            }
        }

        auto output = network.calculateLayer(
                network[HIDDEN],
                layerInput);

        // Now re-remember the newly calculated hidden layer results.
        // We can throw away the result since these are just the old
        // values we already retrieved above:

        network[CONTEXT].activate(output);

        // Finally, calculate the output:

        layerInput = network.calculateLayerTransition(
                network[HIDDEN],
                network[OUTPUT],
                output);
        return network.calculateLayer(
                network[OUTPUT],
                layerInput);
    }
} // namespace wzann


WZANN_REGISTER_CLASS(
        wzann::ElmanNetworkPattern,
        wzann::NeuralNetworkPattern)
