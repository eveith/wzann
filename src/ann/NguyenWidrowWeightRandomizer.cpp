#include <QObject>

#include <cmath>
#include <limits>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "ActivationFunction.h"
#include "Connection.h"

#include "NguyenWidrowWeightRandomizer.h"

#include <iostream>
using std::cout;
using std::endl;

using std::pow;


namespace Winzent {
    namespace ANN {


        NguyenWidrowWeightRandomizer::NguyenWidrowWeightRandomizer(
                QObject *parent):
                    QObject(parent)
        {
        }


        void NguyenWidrowWeightRandomizer::randomize(
                NeuralNetwork *const &network)
                const
        {
            for (int i = 0; i != network->size() - 1; ++i) {
                randomizeSynapse(
                        network,
                        network->layerAt(i),
                        network->layerAt(i+1));
            }
        }


        void NguyenWidrowWeightRandomizer::randomizeSynapse(
                NeuralNetwork *const &network,
                Layer *const &from,
                Layer *const &to)
                const
        {
            int fromCount   = from->size();
            int toCount     = to->size();

            for (int i = 0; i != fromCount; ++i) {
                Neuron *neuron = from->neuronAt(i);

                foreach (Connection *connection,
                         network->neuronConnectionsFrom(neuron)) {
                    if (connection->fixedWeight()) {
                        continue;
                    }

                    if (!to->contains(connection->destination())) {
                        continue;
                    }

                    qreal high = connection->destination()
                            ->activationFunction()
                            ->calculate(std::numeric_limits<qreal>::max());
                    qreal low = connection->destination()
                            ->activationFunction()
                            ->calculate(std::numeric_limits<qreal>::min());
                    qreal b = 0.7 * pow(toCount, (1.d/fromCount)) / (high-low);

                    connection->weight(-b
                            + qrand() / static_cast<qreal>(RAND_MAX) * 2 * b);
                }
            }
        }
    }
}
