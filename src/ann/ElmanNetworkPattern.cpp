#include <initializer_list>
#include <QVector>
#include <QList>

#include <ClassRegistry.h>

#include "Layer.h"
#include "Neuron.h"
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
            // Make sure that we do not get more than three layers here:

            if (layerSizes.size() != 3) {
                throw LayerSizeMismatchException(layerSizes.size(), 3);
            }

            // We need to insert the context layer after the input layer.

            m_layerSizes.insert(CONTEXT, layerSizes.at(1));
            m_activationFunctions.insert(
                    CONTEXT,
                    new RememberingActivationFunction(0.5));
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

            layerSizes.removeAt(CONTEXT);
            activationFunctions.removeAt(CONTEXT);

            return new ElmanNetworkPattern(
                        layerSizes,
                        activationFunctions);
        }


        void ElmanNetworkPattern::configureNetwork(
                NeuralNetwork* const& network)
        {
            // Create layers & neurons:

            for (int lidx = INPUT; lidx <= OUTPUT; ++lidx) {
                Layer *layer = new Layer();
                int layerSize = m_layerSizes.at(lidx);

                for (int i = 0; i != layerSize; ++i) {
                    layer->addNeuron(
                            new Neuron(m_activationFunctions[lidx]->clone()));
                }

                *network << layer;
            }

            // Set up connections:

            for (int lidx = INPUT; lidx < OUTPUT; ++lidx) {
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
                            network->connectNeurons(
                                    network->layerAt(HIDDEN)->neuronAt(i),
                                    network->layerAt(CONTEXT)->neuronAt(i));
                            Connection* connection = network->neuronConnection(
                                    network->layerAt(HIDDEN)->neuronAt(i),
                                    network->layerAt(CONTEXT)->neuronAt(i));
                            connection->weight(1.0);
                            connection->fixedWeight(true);
                        }

                        break;
                    }
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
            auto layerInput = network[INPUT].activate(input);
            layerInput = network.calculateLayerTransition(
                    network[INPUT],
                    network[HIDDEN],
                    layerInput);

            // Fetch remembered values from the context layer:

            {
                auto &contextLayer = network[CONTEXT];
                Vector rememberedValues;
                rememberedValues.reserve(contextLayer.size());

                for (size_t i = 0; i != contextLayer.size(); ++i) {
                    rememberedValues.push_back(contextLayer[i].activate(0.0));
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

            auto output = network[HIDDEN].activate(layerInput);

            // Now re-remember the newly calculated hidden layer results.
            // We can throw away the result since these are just the old
            // values we already retrieved above:

            network[CONTEXT].activate(output);

            // Finally, calculate the output:

            layerInput = network.calculateLayerTransition(
                    network[HIDDEN],
                    network[OUTPUT],
                    output);
            output = network[OUTPUT].activate(layerInput);

            return output;
        }
    }
}


WINZENT_REGISTER_CLASS(
        Winzent::ANN::ElmanNetworkPattern,
        Winzent::ANN::NeuralNetworkPattern)
