#include <numeric>

#include "Vector.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"

#include "GradientAnalysisHelper.h"

namespace Winzent {
    namespace ANN {
        GradientAnalysisHelper::GradientAnalysisHelper()
        {
        }


        GradientAnalysisHelper::~GradientAnalysisHelper()
        {
        }


        double GradientAnalysisHelper::outputNeuronDelta(
                Neuron const& neuron,
                double error)
        {
            return error * calculateDerivative(
                    neuron.activationFunction(),
                    neuron.lastInput());
        }


        double GradientAnalysisHelper::hiddenNeuronDelta(
                NeuralNetwork& ann,
                Neuron const& neuron,
                NeuronDeltaMap& neuronDeltas,
                Vector const& outputError)
        {

            auto connections = ann.connectionsFrom(neuron);
            assert(connections.second - connections.first > 0);


            return calculateDerivative(
                    neuron.activationFunction(),
                    neuron.lastInput())
                * std::accumulate(
                    connections.first,
                    connections.second,
                    0.0,
                    [&ann, &neuronDeltas, &outputError](
                        double const& delta,
                        Connection* const& c) {
                return delta + c->weight() * neuronDelta(
                        ann,
                        c->destination(),
                        neuronDeltas,
                        outputError);
            });
        }


        double GradientAnalysisHelper::neuronDelta(
                NeuralNetwork& ann,
                Neuron const& neuron,
                NeuronDeltaMap& neuronDeltas,
                Vector const& outputError)
        {
            assert(! ann.inputLayer().contains(neuron));

            if (neuronDeltas.find(&neuron) != neuronDeltas.end()) {
                return neuronDeltas.at(&neuron);
            }

            if (ann.outputLayer().contains(neuron)) {
                auto neuronIdx = ann.outputLayer().indexOf(neuron);
                neuronDeltas[&neuron] = outputNeuronDelta(
                        neuron,
                        outputError.at(neuronIdx));
            } else {
                neuronDeltas[&neuron] = hiddenNeuronDelta(
                        ann,
                        neuron,
                        neuronDeltas,
                        outputError);
            }

            return neuronDeltas.at(&neuron);
        }
    } // namespace training
} // namespace wzann
