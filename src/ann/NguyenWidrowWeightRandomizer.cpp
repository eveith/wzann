#include <cmath>
#include <limits>
#include <cstddef>

#include <boost/random.hpp>

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"

#include "WeightRandomizer.h"
#include "NguyenWidrowWeightRandomizer.h"


using std::pow;


namespace Winzent {
    namespace ANN {
        NguyenWidrowWeightRandomizer::NguyenWidrowWeightRandomizer():
                WeightRandomizer()
        {
        }


        void NguyenWidrowWeightRandomizer::randomize(NeuralNetwork &network)
        {
            for (size_t i = 0; i != network.size() - 1; ++i) {
                randomizeSynapse(
                        network,
                        *(network.layerAt(i)),
                        *(network.layerAt(i+1)));
            }
        }


        void NguyenWidrowWeightRandomizer::randomizeSynapse(
                NeuralNetwork &network,
                Layer &from,
                Layer &to)
                const
        {
            boost::random::mt11213b rng;
            boost::random::uniform_01<qreal> rDistribution;
            auto fromCount   = from.size();
            auto toCount     = to.size();

            for (size_t i = 0; i != fromCount; ++i) {
                Neuron *neuron = from.neuronAt(i);

                for (auto &connection:
                        network.neuronConnectionsFrom(neuron)) {
                    if (connection->fixedWeight()) {
                        continue;
                    }

                    if (! to.contains(connection->destination())) {
                        continue;
                    }

                    qreal high = connection->destination()
                            ->activationFunction()
                            ->calculate(std::numeric_limits<qreal>::max());
                    qreal low = connection->destination()
                            ->activationFunction()
                            ->calculate(std::numeric_limits<qreal>::min());
                    qreal b = pow(
                                toCount,
                                (1.0 / static_cast<qreal>(fromCount)))
                            / (high-low) * 0.7;

                    connection->weight(-b + rDistribution(rng) * 2 * b);
                }
            }
        }
    }
}
