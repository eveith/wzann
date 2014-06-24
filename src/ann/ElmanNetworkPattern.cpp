/*!
 * \file	ElmanNetworkPattern.cpp
 * \brief
 * \date	31.12.2012
 * \author	eveith
 */


#include <initializer_list>
#include <QVector>
#include <QList>

#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "RememberingActivationFunction.h"
#include "Layer.h"
#include "Connection.h"
#include "Neuron.h"
#include "Exception.h"

#include "ElmanNetworkPattern.h"


using std::initializer_list;


namespace Winzent
{
    namespace ANN
    {
        ElmanNetworkPattern::ElmanNetworkPattern(
                QList<int> layerSizes,
                QList<ActivationFunction*> activationFunctions,
                QObject* parent):
                    NeuralNetworkPattern(layerSizes,
                                activationFunctions,
                                parent)
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
                initializer_list<ActivationFunction*> activationFunctions,
                QObject *parent):
                    ElmanNetworkPattern(layerSizes, activationFunctions, parent)
        {
        }


        NeuralNetworkPattern* ElmanNetworkPattern::clone() const
        {
            QList<int> layerSizes = m_layerSizes;
            QList<ActivationFunction *> activationFunctions;

            foreach (ActivationFunction *f, m_activationFunctions) {
                activationFunctions << f->clone();
            }

            // Delete the CONTEXT layer, as the constructor will try to add it
            // and fail if the network has more than three layers:

            layerSizes.removeAt(CONTEXT);
            activationFunctions.removeAt(CONTEXT);

            return new ElmanNetworkPattern(
                        layerSizes,
                        activationFunctions,
                        parent());
        }


        void ElmanNetworkPattern::configureNetwork(NeuralNetwork *network)
        {
            // Create layers & neurons:

            for (int lidx = INPUT; lidx <= OUTPUT; ++lidx) {

                Layer *layer = new Layer(network);
                int layerSize = m_layerSizes.at(lidx);

                for (int i = 0; i != layerSize; ++i) {
                    *layer << new Neuron(m_activationFunctions[lidx]->clone());
                }

                *network << layer;
            }

            // Set up connections:

            for (int lidx = INPUT; lidx < OUTPUT; ++lidx) {
                int layerSize = m_layerSizes.at(lidx);

                switch (lidx) {
                    case INPUT: {
                        fullyConnectNetworkLayers(network, lidx, HIDDEN);

                        for (int i = 0; i != m_layerSizes.at(lidx); ++i) {
                            for (int j = 0; j != m_layerSizes.at(HIDDEN); ++j) {
                                network->neuronConnection(
                                        network->layerAt(lidx)->neuronAt(i),
                                        network->layerAt(HIDDEN)->neuronAt(j)
                                )->setRandomWeight(
                                        m_weightRandomMin, m_weightRandomMax);
                            }
                        }

                        break;
                    }
                    case CONTEXT: {
                        fullyConnectNetworkLayers(network, lidx, HIDDEN);

                        for (int i = 0; i != m_layerSizes.at(lidx); ++i) {
                            for (int j = 0; j != m_layerSizes.at(HIDDEN); ++j) {
                                Connection *c = network->neuronConnection(
                                        network->layerAt(lidx)->neuronAt(i),
                                        network->layerAt(HIDDEN)->neuronAt(j));
                                c->setRandomWeight(
                                        m_weightRandomMin, m_weightRandomMax);
                            }
                        }

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

                        for (int i = 0; i != m_layerSizes.at(lidx); ++i) {
                            for (int j = 0; j != m_layerSizes.at(OUTPUT); ++j) {
                                network->neuronConnection(
                                        network->layerAt(lidx)->neuronAt(i),
                                        network->layerAt(OUTPUT)->neuronAt(j)
                                )->setRandomWeight(
                                        m_weightRandomMin, m_weightRandomMax);
                            }
                        }

                        break;
                    }
                }
            }
        }


        ValueVector ElmanNetworkPattern::calculate(
                NeuralNetwork *const &network,
                const ValueVector &input)
        {
            ValueVector layerInput;
            ValueVector output;

            layerInput = network->calculateLayer(INPUT, input);
            layerInput = network->calculateLayerTransition(
                    INPUT,
                    HIDDEN,
                    layerInput);

            // Fetch remembered values from the context layer:

            {
                Layer* contextLayer = (*network)[CONTEXT];
                ValueVector rememberedValues(contextLayer->size());

                for (int i = 0; i != contextLayer->size(); ++i) {
                    rememberedValues[i] = contextLayer->neuronAt(i)
                            ->activate(0.0);
                }

                rememberedValues = network->calculateLayerTransition(
                        CONTEXT,
                        HIDDEN,
                        rememberedValues);

                for (int i = 0; i != rememberedValues.size(); ++i) {
                    layerInput[i] += rememberedValues[i];
                }
            }

            output = network->calculateLayer(HIDDEN, layerInput);

            // Now re-remember the newly calculated hidden layer results. We can
            // throw away the result since these are just the old values we already
            // retrieved above:

            network->calculateLayer(CONTEXT, output);

            // Finally, calculate the output:

            layerInput = network->calculateLayerTransition(
                    HIDDEN,
                    OUTPUT,
                    output);
            output = network->calculateLayer(OUTPUT, layerInput);

            return output;
        }
    }
}
