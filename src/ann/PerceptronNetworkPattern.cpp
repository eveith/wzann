#include <initializer_list>

#include <QList>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "ActivationFunction.h"
#include "NeuralNetworkPattern.h"
#include "NguyenWidrowWeightRandomizer.h"

#include "PerceptronNetworkPattern.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {

        PerceptronNetworkPattern::PerceptronNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction *> activationFunctions,
                QObject *parent):
                    NeuralNetworkPattern(
                        layerSizes,
                        activationFunctions,
                        parent)
        {
        }


        PerceptronNetworkPattern::PerceptronNetworkPattern(
                initializer_list<int> layerSizes,
                initializer_list<ActivationFunction *> activationFunctions,
                QObject *parent):
                    PerceptronNetworkPattern(
                        QList<int>(layerSizes),
                        QList<ActivationFunction*>(activationFunctions),
                        parent)
        {
        }


        NeuralNetworkPattern *PerceptronNetworkPattern::clone() const
        {
            return new PerceptronNetworkPattern(
                    QList<int>(m_layerSizes),
                    QList<ActivationFunction *>(m_activationFunctions),
                    parent());
        }


        void PerceptronNetworkPattern::configureNetwork(
                NeuralNetwork *network)
        {
            // Add the layers & neurons:

            for (int i = 0; i != m_layerSizes.size(); ++i) {
                Layer *layer = new Layer(network);
                ActivationFunction *activationFunction =
                        m_activationFunctions.at(i)->clone();

                int size = m_layerSizes.at(i);
                for (int j = 0; j != size; ++j) {
                    layer->addNeuron(new Neuron(activationFunction));
                }

                *network << layer;
            }

            // Now connect layers:

            for (int i = 0; i != network->size() -1; ++i) {
                fullyConnectNetworkLayers(network, i, i+1);
            }

            // Init weights:

            NguyenWidrowWeightRandomizer().randomize(network);
        }


        ValueVector PerceptronNetworkPattern::calculate(
                NeuralNetwork *const &network,
                const ValueVector &input)
        {
            ValueVector output = input; // For the loop

            for (int i = 0; i != network->size(); ++i) {
                output = network->calculateLayer(network->layerAt(i), output);

                if (i < network->size() - 1) {
                    output = network->calculateLayerTransition(i, i+1, output);
                }
            }

            return output;
        }
    } // namespace ANN
} // namespace Winzent
