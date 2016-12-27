#include <initializer_list>
#include <QVector>
#include <QList>

#include <ClassRegistry.h>

#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Exception.h"
#include "Connection.h"
#include "NeuralNetwork.h"

#include "ActivationFunction.h"
#include "RememberingActivationFunction.h"

#include "NeuralNetworkPattern.h"
#include "ElmanNetworkPattern.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {
        ElmanNetworkPattern::ElmanNetworkPattern()
        {
        }


        ElmanNetworkPattern::ElmanNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction *> activationFunctions):
                    NeuralNetworkPattern(layerSizes, activationFunctions)
        {
        }


        ElmanNetworkPattern::ElmanNetworkPattern(
                initializer_list<int> layerSizes,
                initializer_list<ActivationFunction*> activationFunctions):
                    ElmanNetworkPattern(
                        QList<int>(layerSizes), // Cast necessary!
                        QList<ActivationFunction *>(activationFunctions))
        {
        }


        NeuralNetworkPattern* ElmanNetworkPattern::clone() const
        {
            QList<int> layerSizes = m_layerSizes;
            QList<ActivationFunction *> activationFunctions;

            for (const auto &f: m_activationFunctions) {
                activationFunctions.push_back(f->clone());
            }

            // Delete the CONTEXT layer, as the constructor will try to add it
            // and fail if the network has more than three layers:

            if (layerSizes.size() > 3) {
                layerSizes.removeAt(CONTEXT);
                activationFunctions.removeAt(CONTEXT);
            }

            return new ElmanNetworkPattern(
                        layerSizes,
                        activationFunctions);
        }


        void ElmanNetworkPattern::configureNetwork(NeuralNetwork &network)
        {

            // Make sure that we do not get more than three layers here:

            if (m_layerSizes.size() != 3) {
                throw LayerSizeMismatchException(m_layerSizes.size(), 3);
            }

            // We need to insert the context layer after the input layer.

            m_layerSizes.insert(CONTEXT, m_layerSizes.at(1));
            m_activationFunctions.insert(
                    CONTEXT,
                    new RememberingActivationFunction(1.0));

            // Create layers & neurons:

            for (int lidx = INPUT; lidx <= OUTPUT; ++lidx) {
                Layer *layer = new Layer();
                int layerSize = m_layerSizes.at(lidx);

                for (int i = 0; i != layerSize; ++i) {
                    layer->addNeuron(
                            new Neuron(m_activationFunctions[lidx]->clone()));
                }

                network << layer;
            }

            // Set up connections:

            for (int lidx = INPUT; lidx <= OUTPUT; ++lidx) {
                int layerSize = m_layerSizes.at(lidx);

                switch (lidx) {
                case INPUT: {
                    fullyConnectNetworkLayers(network, lidx, HIDDEN);
                    break;
                }
                case CONTEXT: {
                    fullyConnectNetworkLayers(network, lidx, HIDDEN);
                    break;
                }
                case HIDDEN: {
                    fullyConnectNetworkLayers(network, lidx, OUTPUT);

                    for (int i = 0; i != layerSize; ++i) {
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


        bool ElmanNetworkPattern::equals(
                const NeuralNetworkPattern* const &other)
                const
        {
            return reinterpret_cast<const ElmanNetworkPattern* const&>(
                        other) != nullptr
                    && NeuralNetworkPattern::equals(other);
        }


        Vector ElmanNetworkPattern::calculate(
                NeuralNetwork &network,
                const Vector &input)
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
    }
}


WINZENT_REGISTER_CLASS(
        Winzent::ANN::ElmanNetworkPattern,
        Winzent::ANN::NeuralNetworkPattern)
