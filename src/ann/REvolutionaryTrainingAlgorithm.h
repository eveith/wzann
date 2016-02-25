/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de>
 * \date 2013-03-19
 */


#ifndef WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
#define WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H


#include <QObject>

#include <cstddef>
#include <ostream>

#include <log4cxx/logger.h>

#include <boost/random.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "REvol.h"

#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"

#include "Winzent-ANN_global.h"


using std::size_t;
using std::ptrdiff_t;


namespace Winzent {
    namespace ANN {


        class Weight;
        class TrainingSet;


        /*!
         * \brief The Individual class represents an individual during the
         *  training phase of the evolutionary algorithm
         */
        struct Individual: public Winzent::Algorithm::detail::Individual
        {
            /*!
             * \brief Constructs an empty Individual
             *
             * Initializes the Individual's TTL to 0.
             */
            Individual();


            /*!
             * \brief Copy constructor for the base class
             *
             * \param[in] individual The individual to copy
             */
            Individual(const Algorithm::detail::Individual &individual);


            /*!
             * \brief Creates a new individual given a neural network
             *
             * \param[in] neuralNetwork The ANN from which we initialize
             *  the parameters and scatter vector
             */
            Individual(const NeuralNetwork &neuralNetwork);


            /*!
             * \brief Retrieves parameters from the supplied ANN
             *
             * \param[in] neuralNetwork The ANN from which the parameters
             *  shall be retrieved
             *
             * \return The parameters vector
             */
            static Vector parametersFromNeuralNetwork(
                    const NeuralNetwork &neuralNetwork);


            /*!
             * \brief Applies the parameters of this individual to
             *  the supplied ANN
             *
             * \param[in] The individual from which that parameters are taken
             *
             * \param[inout] neuralNetwork The Artificial Neural Network to
             *  which the parameters stored in the Individual shall be applied
             *
             * \return `*this`
             */
            static void applyParameters(
                    const Individual &individual,
                    NeuralNetwork &neuralNetwork);


            /*!
             * \brief Allows access to the error vector
             *
             * \return A modifiable reference to the error vector
             */
            Vector &errorVector();


            /*!
             * \brief Returns a read-only copy of the error vector
             *
             * \return The error vector write-protected
             */
            const Vector &errorVector() const;
        };


        class WINZENTANNSHARED_EXPORT REvolutionaryTrainingAlgorithm:
                public TrainingAlgorithm,
                public Winzent::Algorithm::REvol
        {
        public:


            /*!
             * \brief Creates a new instance of the
             *  evolutionary training algorithm for
             *  training a particular network.
             */
            REvolutionaryTrainingAlgorithm();


            /*!
             * \brief Generates the initial population from the supplied base
             *  network
             *
             * \param[in] baseNetwork The base network the user supplied for
             *  training
             *
             * \return The population, including the elite
             */
            Population generateInitialPopulation(
                    const NeuralNetwork &baseNetwork);


            /*!
             * \brief Evaluates one individual
             *
             * This method uses the given training set to evaluate the
             * indicited individual. It feeds the patterns to the individual,
             * notes the errors, and calculates the mean squared error (MSE).
             *
             * \param[in] individual The individual. Its error vector must
             *  have the size necessary to store the MSE and all error values.
             *
             * \param[in] ann The Artificial Neural Network the individual
             *  applies to
             *
             * \param[in] trainingSet The training set that should be used to
             *  evaluate the individual
             */
            void evaluateIndividual(
                    Algorithm::detail::Individual &individual,
                    NeuralNetwork& ann,
                    const TrainingSet &trainingSet);



            /*!
             * \brief Trains the Neural Network using Ruppert's evolutionary
             *  training algorithm.
             *
             * \param trainingSet
             */
            virtual void train(NeuralNetwork &ann, TrainingSet &trainingSet)
                    override;
        };
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::REvolutionaryTrainingAlgorithm &algorithm);
}


#endif // WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
