#include <initializer_list>

#include <QList>

#include <ClassRegistry.h>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"

#include "PerceptronNetworkPattern.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {
        PerceptronNetworkPattern::PerceptronNetworkPattern()
        {
        }


        PerceptronNetworkPattern::PerceptronNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction *> activationFunctions):
                    NeuralNetworkPattern(layerSizes, activationFunctions)
        {
        }


        PerceptronNetworkPattern::PerceptronNetworkPattern(
                initializer_list<int> layerSizes,
                initializer_list<ActivationFunction *> activationFunctions):
                    PerceptronNetworkPattern(
                        QList<int>(layerSizes),
                        QList<ActivationFunction *>(activationFunctions))
        {
        }


        NeuralNetworkPattern *PerceptronNetworkPattern::clone() const
        {
            QList<ActivationFunction *> functionClones;

            for (const auto &i: m_activationFunctions) {
                functionClones.push_back(i->clone());
            }

            return new PerceptronNetworkPattern(
                    m_layerSizes,
                    functionClones);
        }


        bool PerceptronNetworkPattern::equals(
                const NeuralNetworkPattern* const& other)
                const
        {
            return reinterpret_cast<const PerceptronNetworkPattern* const&>(
                        other) != nullptr
                    && NeuralNetworkPattern::equals(other);
        }


        void PerceptronNetworkPattern::configureNetwork(
                NeuralNetwork* const& network)
        {
            // Add the layers & neurons:

            for (int i = 0; i != m_layerSizes.size(); ++i) {
                Layer *layer = new Layer();
                std::shared_ptr<ActivationFunction> af(
                        m_activationFunctions.at(i)->clone());

                int size = m_layerSizes.at(i);
                for (int j = 0; j != size; ++j) {
                    layer->addNeuron(new Neuron(af));
                }

                *network << layer;
            }

            // Now connect layers:

            for (size_t i = 0; i != network->size(); ++i) {
                if (i > 0) {
                    for (auto &neuron: (*network)[i]) {
                        network->connectNeurons(
                                network->biasNeuron(),
                                &neuron)
                            .weight(-1.0)
                            .fixedWeight(false);
                    }
                }

                if (i + 1 < network->size()) {
                    fullyConnectNetworkLayers(network, i, i+1);
                }
            }
        }


        Vector PerceptronNetworkPattern::calculate(
                NeuralNetwork &network,
                const Vector &input)
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
