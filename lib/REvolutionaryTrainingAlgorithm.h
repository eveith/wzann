#ifndef WZANN_EVOLUTIONARYTRAININGALGORITHM_H_
#define WZANN_EVOLUTIONARYTRAININGALGORITHM_H_


#include <cstddef>
#include <ostream>

#include <boost/random.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <wzalgorithm/REvol.h>
#include <wzalgorithm/config.h>

#include "TrainingAlgorithm.h"


namespace wzann {
    class TrainingSet;
    class NeuralNetwork;


    class REvolutionaryTrainingAlgorithm :
            public TrainingAlgorithm,
            public wzalgorithm::REvol
    {
    public:


        /*!
         * \brief Creates a new instance of the multi-part evolutionary
         *  training algorithm for training artificial neural networks.
         */
        REvolutionaryTrainingAlgorithm();


        /*!
         * \brief Reads the current weight vector of an artificial neural
         *  network and writes it to the supplied paramter vector
         *
         * \param[in] ann The artificial neural network to read the parameters
         *  from
         *
         * \param[inout] parameters The vector to write the weight values to;
         *  assumes an empty parameters vector
         */
        static void getWeights(
                NeuralNetwork const& ann,
                wzalgorithm::vector_t& parameters);


        /*!
         * \brief Applies an individual's parameter vector to the weights of
         *  an artificial neural network
         *
         * \param[in] parameters The individual's parameter vector
         *
         * \param[inout] ann The artificial neural network to apply the
         *  parameter vector to
         *
         */
        static void applyParameters(
                wzalgorithm::vector_t const& parameters,
                NeuralNetwork& ann);


        /*!
         * \brief Evaluates one individual
         *
         * This method uses the given training set to evaluate the
         * indicited individual. It feeds the patterns to the individual,
         * notes the errors, and calculates the training error. The training
         * error is saved in the
         *
         * \param[inout] individual The individual: its parameter vector is
         *  applied to the artificial neural network's connections, and the
         *  training set is then run in order to calculate the error value.
         *
         * \param[in] ann The Artificial Neural Network the individual
         *  applies to
         *
         * \param[in] trainingSet The training set that should be used to
         *  evaluate the individual
         *
         * \return `true` if the current individual satisfies the target
         *  error set in the trainingSet, `false` otherwise.
         */
        static bool individualSucceeds(
                wzalgorithm::REvol::Individual& individual,
                NeuralNetwork& ann,
                TrainingSet const& trainingSet);


        /*!
         * \brief Trains the Neural Network using Ruppert's evolutionary
         *  training algorithm.
         *
         * \param trainingSet
         */
        virtual void train(NeuralNetwork& ann, TrainingSet& trainingSet)
                override;
    };
} // namespace wzann


namespace std {
    ostream &operator<<(
            ostream& os,
            wzann::REvolutionaryTrainingAlgorithm const& algorithm);
}


#endif // WZANN_EVOLUTIONARYTRAININGALGORITHM_H_
