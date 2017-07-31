#include <cmath>
#include <limits>
#include <numeric>

#include <boost/range.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Connection.h"
#include "TrainingSet.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "GradientAnalysisHelper.h"

#include "BackpropagationTrainingAlgorithm.h"


using std::pow;
using std::accumulate;

using boost::adaptors::reverse;
using boost::make_iterator_range;


namespace wzann {
    BackpropagationTrainingAlgorithm::BackpropagationTrainingAlgorithm() :
            TrainingAlgorithm(),
            m_learningRate(DEFAULT_LEARNING_RATE)
    {
    }


    double BackpropagationTrainingAlgorithm::learningRate() const
    {
        return m_learningRate;
    }


    BackpropagationTrainingAlgorithm&
    BackpropagationTrainingAlgorithm::learningRate(double rate)
    {
        m_learningRate = rate;
        return *this;
    }


    void BackpropagationTrainingAlgorithm::train(
            NeuralNetwork& ann,
            TrainingSet& trainingSet)
    {
        // Initialize the state variables:

        size_t epochs = 0;
        double error = std::numeric_limits<double>::max();

        for(; epochs < trainingSet.maxEpochs()
                    && error > trainingSet.targetError();
                ++epochs) {
            error = 0.0;
            size_t numRelevantItems = 0;

            for (auto const& ti: trainingSet.trainingItems) {
                GradientAnalysisHelper::NeuronDeltaMap neuronDeltas;
                ConnectionDeltaMap connectionDeltas;

                // First step: Feed forward and compare the network's
                // output with the ideal teaching output:

                const Vector actualOutput = ann.calculate(ti.input());
                if (! ti.outputRelevant()) {
                    continue;
                }

                numRelevantItems++;
                Vector errorOutput;
                errorOutput.reserve(actualOutput.size());
                error += GradientAnalysisHelper::errors(
                        make_iterator_range(actualOutput),
                        make_iterator_range(ti.expectedOutput()),
                        std::back_inserter(errorOutput));

                // Propagate the error backwards:

                for (auto* c: reverse(ann.connections())) {
                    connectionDeltas[c] =
                            GradientAnalysisHelper::neuronDelta(
                                ann,
                                c->destination(),
                                neuronDeltas,
                                errorOutput);
                }

                // Apply calculated delta values:

                for (auto& cd: connectionDeltas) {
                    auto* connection = cd.first;
                    auto delta = cd.second;

                    connection->weight(connection->weight()
                            - (learningRate() * delta
                               * connection->source().lastResult()));
                }
            }

            // It's called MEAN square error for a reason:

            error /= numRelevantItems;
        }

        // Store final training results:

        setFinalError(trainingSet, error);
        setFinalNumEpochs(trainingSet, epochs);
    }
} // namespace wzann
