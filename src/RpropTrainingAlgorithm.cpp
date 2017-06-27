#include <cmath>
#include <limits>
#include <algorithm>

#include <boost/range.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "TrainingSet.h"
#include "NeuralNetwork.h"
#include "GradientAnalysis.h"
#include "ActivationFunction.h"

#include "RpropTrainingAlgorithm.h"


using std::fabs;
using std::max;
using std::min;
using std::accumulate;

using boost::adaptors::reverse;
using boost::make_iterator_range;


namespace Winzent {
    namespace ANN {


        const double RpropTrainingAlgorithm::ETA_POSITIVE =  1.2;
        const double RpropTrainingAlgorithm::ETA_NEGATIVE = -0.5;
        const double RpropTrainingAlgorithm::ZERO_TOLERANCE =
                0.00000000000000001;
        const double RpropTrainingAlgorithm::DEFAULT_INITIAL_UPDATE = 0.1;
        const double RpropTrainingAlgorithm::DELTA_MIN = 1e-6;
        const double RpropTrainingAlgorithm::MAX_STEP = 50.0;



        RpropTrainingAlgorithm::RpropTrainingAlgorithm() :
                TrainingAlgorithm()
        {
        }


        int RpropTrainingAlgorithm::sgn(double x)
        {
            if (fabs(x) < ZERO_TOLERANCE) {
                return 0;
            }

            if (x < 0.0) {
                return -1;
            }

            return 1;
        }


        void RpropTrainingAlgorithm::train(
                NeuralNetwork &ann,
                TrainingSet &trainingSet)
        {
            ConnectionGradientMap currentGradients;
            ConnectionGradientMap lastGradients;
            ConnectionGradientMap updateValues;
            ConnectionGradientMap lastWeightChange;
            double error = std::numeric_limits<double>::max();
            size_t epoch = 0;

            for(; epoch < trainingSet.maxEpochs()
                        && error > trainingSet.targetError();
                    ++epoch) {
                error = 0.0;
                currentGradients.clear();
                size_t numRelevantItems = 0;

                // Forward pass:

                for (auto const& ti: trainingSet.trainingItems) {
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

                    GradientAnalysisHelper::NeuronDeltaMap neuronDeltas;

                    // Calculate error delta of all neurons
                    // in the forward pass:

                    for (auto* c: reverse(ann.connections())) {
                        if (c->fixedWeight()) {
                            continue;
                        }

                        auto const& dstNeuron = c->destination();
                        auto delta = GradientAnalysisHelper::neuronDelta(
                                ann,
                                c->destination(),
                                neuronDeltas,
                                errorOutput);
                        neuronDeltas[&dstNeuron] = delta;

                        // Add up gradients. The default-constructed value
                        // in for value-initalization is zero initialization:

                        currentGradients[c] += delta*c->source().lastResult();
                    }
                }

                // Calculate mean of all errors:

                error /= numRelevantItems;

                // Now, learn:

                for (auto const& gradient: currentGradients) {
                    auto* c = gradient.first;

                    int change = sgn(currentGradients[c] * lastGradients[c]);
                    double dw = 0.0;

                    double updateValue = DEFAULT_INITIAL_UPDATE;
                    if (updateValues.find(c) != updateValues.end()) {
                        updateValue = updateValues[c];
                    }

                    if (0 == change) {
                        double delta = updateValue;
                        dw = sgn(currentGradients[c]) * delta;
                        lastGradients[c] = currentGradients[c];
                    } else if (change > 0) { // Retained sign, increase step:
                        double delta = updateValue * ETA_POSITIVE;
                        delta = min(delta, MAX_STEP);
                        dw = sgn(currentGradients[c]) * delta;
                        updateValues[c] = delta;
                        lastGradients[c] = currentGradients[c];
                        lastWeightChange[c] = dw;
                    } else { // change < 0 --- Last delta was too big
                        double delta = updateValue * ETA_NEGATIVE;
                        delta = max(delta, DELTA_MIN);
                        updateValues[c] = delta;
                        dw = -lastWeightChange[c];

                        // Set the previous gradent to zero so that there will
                        // be no adjustment the next iteration:

                        lastGradients[c] = 0.0;
                    }

                    c->weight(c->weight() - dw);
                }
            }

            setFinalError(trainingSet, error);
            setFinalNumEpochs(trainingSet, epoch);
        }
    } // namespace ANN
} // namespace Winzent
