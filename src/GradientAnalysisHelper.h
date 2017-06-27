#ifndef GRADIENTANALYSIS_H
#define GRADIENTANALYSIS_H


#include <cmath>
#include <unordered_map>

#include <boost/range/iterator_range_core.hpp>

#include "Vector.h"


namespace Winzent {
    namespace ANN {


        class Neuron;
        class NeuralNetwork;


        /*!
         * \brief The GradientAnalysis class offers `neuron delta' calculation
         *  functions usable in gradient descent algorithms.
         */
        class GradientAnalysisHelper
        {
        public:

            typedef std::unordered_map<
                    Neuron const*,
                    double> NeuronDeltaMap;

            GradientAnalysisHelper();
            virtual ~GradientAnalysisHelper();


            /*!
             * \brief The backpropagation error function
             *
             * This method implements both the error function of
             * the backpropagation algorithm and its derivative. The error
             * funciton is: \f[
             * E = \frac{1}{2} (\mathit{expected} - \mathit{actual})^2~.
             * \f]
             *
             * Thus, its derivative becomes: \f[
             * E' = \mathit{target} - \mathit{expected}~.
             * \f]
             *
             * The result of the derivateive is written to the output iterator
             * `out`, while the result of the sum of the application of the
             * (original) error function is given as return value.
             *
             * \param[in] actual Iterator range of the values that are
             *  returned by the ANN
             *
             * \param[in] expected Iterator range containing the expected
             *  values
             *
             * \param[out] out The output iterator where the piece-wise
             *  calculation of \f$\mathit{actual} - \mathit{expected}\f$ is
             *  written to.
             *
             * \returns \f$\frac{1}{2}\sum_i (
             *  \mathit{expected}_i - \mathit{actual}_i)^2\f$
             */
            template <class ForwardTraversalIterator, class OutIter>
            static double errors(
                    boost::iterator_range<ForwardTraversalIterator> actual,
                    boost::iterator_range<ForwardTraversalIterator> expected,
                    OutIter out)
            {
                double error = 0.0;

                for (auto ait = actual.begin(), eit = expected.begin();
                        ait != actual.end() && eit != expected.end();
                        ait++, eit++, out++) {
                    *out = *ait - *eit;
                    error += std::pow(*eit - *ait, 2);
                }

                return error / 2.0;
            }


            /*!
             * \brief Calculates the neuron delta for one given neuron,
             *  assuming it is an output layer neuron.
             */
            static double outputNeuronDelta(
                    Neuron const& neuron,
                    double error);


            /*!
             * \brief Calculates the neuron delta for a neuron in an hidden
             *  layer.
             */
            static double hiddenNeuronDelta(
                    NeuralNetwork& ann,
                    Neuron const& neuron,
                    NeuronDeltaMap& neuronDeltas,
                    Vector const& outputError);


            /*!
             * \brief Calculates the delta value of a neuron.
             */
            static double neuronDelta(
                    NeuralNetwork& ann,
                    Neuron const& neuron,
                    NeuronDeltaMap& neuronDeltas,
                    Vector const& outputError);
        };
    } // namespace training
} // namespace wzann

#endif // GRADIENTANALYSIS_H
