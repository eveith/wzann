#include <QtGlobal>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "Connection.h"
#include "NeuralNetwork.h"
#include "SimpleWeightRandomizer.h"


namespace Winzent {
    namespace ANN {
        SimpleWeightRandomizer::SimpleWeightRandomizer():
                m_minWeight(defaultMinWeight),
                m_maxWeight(defaultMaxWeight)
        {
        }


        double SimpleWeightRandomizer::minWeight() const
        {
            return m_minWeight;
        }


        SimpleWeightRandomizer &SimpleWeightRandomizer::minWeight(
                const double &weight)
        {
            m_minWeight = weight;
            return *this;
        }


        double SimpleWeightRandomizer::maxWeight() const
        {
            return m_maxWeight;
        }


        SimpleWeightRandomizer &SimpleWeightRandomizer::maxWeight(
                const double &weight)
        {
            m_maxWeight = weight;
            return *this;
        }


        void SimpleWeightRandomizer::randomize(NeuralNetwork &neuralNetwork)
        {
            boost::random::mt11213b rng;
            boost::random::uniform_real_distribution<double> rDistribution(
                    m_minWeight,
                    m_maxWeight);

            neuralNetwork.eachConnection(
                    [&rng, &rDistribution](Connection *const &c) {
                if (!c->fixedWeight()) {
                    c->weight(rDistribution(rng));
                }
            });
        }
    } // namespace ANN
} // namespace Winzent

