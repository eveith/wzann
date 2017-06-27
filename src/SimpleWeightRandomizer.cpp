#include <boost/range.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "Connection.h"
#include "NeuralNetwork.h"
#include "SimpleWeightRandomizer.h"


using boost::make_iterator_range;


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
                double weight)
        {
            m_minWeight = weight;
            return *this;
        }


        double SimpleWeightRandomizer::maxWeight() const
        {
            return m_maxWeight;
        }


        SimpleWeightRandomizer &SimpleWeightRandomizer::maxWeight(
                double weight)
        {
            m_maxWeight = weight;
            return *this;
        }


        void SimpleWeightRandomizer::randomize(NeuralNetwork& neuralNetwork)
        {
            boost::random::mt11213b rng;
            boost::random::uniform_real_distribution<double> rDistribution(
                    m_minWeight,
                    m_maxWeight);

            for (auto* c: make_iterator_range(neuralNetwork.connections())) {
                if (! c->fixedWeight()) {
                    c->weight(rDistribution(rng));
                }
            }
        }
    } // namespace ANN
} // namespace Winzent

