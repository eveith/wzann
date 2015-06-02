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

#include <boost/random.hpp>

#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class Weight;
        class TrainingSet;


        /*!
         * \brief The Individual class represents an individual during the
         *  training phase of the evolutionary algorithm
         */
        class Individual
        {
        private:


            /*!
             * \brief The parameters vector of this individual
             */
            ValueVector m_parameters;


            /*!
             * \brief The scatter vector of this individual
             */
            ValueVector m_scatter;


            /*!
             * \brief How many iterations this individual may exist
             */
            int m_timeToLive;


            /*!
             * \brief Stores the result of the last runs. Index 0 stores the
             *  overall error, indices 1..n stores the results of the indivdual
             *  runs.
             */
            ValueVector m_errorVector;


        public:


            /*!
             * \brief Constructs an empty Individual
             *
             * Initializes the Individual's TTL to 0.
             */
            Individual();


            /*!
             * \brief Creates a new individual given a neural network
             *
             * \param[in] neuralNetwork The ANN from which we initialize
             *  the parameters and scatter vector
             */
            Individual(const NeuralNetwork &neuralNetwork);


            /*!
             * \brief Constructs a new individual using a parameter Vector
             *
             * \param[in] parameters A parameter vector
             */
            Individual(const ValueVector &parameters);


            /*!
             * \brief Retrieves the current scatter vector
             *
             * \return The current scatter vector
             */
            ValueVector scatter() const;


            /*!
             * \brief Retrieves a modifiable copy of the current
             *  scatter vector.
             *
             * \return The scatter vector, modifiable
             */
            ValueVector &scatter();


            /*!
             * \brief Sets the new scatter vector
             *
             * \param[in] scatter The new scatter vector
             *
             * \return `*this`
             */
            Individual &scatter(ValueVector scatter);


            /*!
             * \brief Retrieves the current parameters set.
             *
             * \return The current parameters
             */
            const ValueVector &parameters() const;


            /*!
             * \brief Allows access to the parameter vector of this Individual
             *
             * \return A modifiable reference to the parameter vector
             */
            ValueVector &parameters();


            /*!
             * \brief Retrieves parameters from the supplied ANN
             *
             * \param[in] neuralNetwork The ANN from which the parameters
             *  shall be retrieved
             *
             * \return The parameters vector
             */
            ValueVector parameters(const NeuralNetwork &neuralNetwork) const;


            /*!
             * \brief Sets new parameters to this individual
             *
             * Sets new parameters to the individual.
             *
             * \param parameters The new set of parameters
             *
             * \return `*this`
             */
            Individual &parameters(const ValueVector &parameters);


            /*!
             * \brief Applies the parameters of this individual to
             *  the supplied ANN
             *
             * \param[inout] neuralNetwork The Artificial Neural Network to
             *  which the parameters stored in the Individual shall be applied
             *
             * \return `*this`
             */
            void applyParameters(NeuralNetwork &neuralNetwork) const;


            /*!
             * \brief Returns the number of iterations this individual may life
             *
             * \return The individual's time to live.
             */
            int timeToLive() const;


            /*!
             * \brief Sets the individual's new time to live
             *
             * \param[in] ttl The number of iterations this individual
             *  participates in the training
             *
             * \return `*this`
             */
            Individual &timeToLive(const int &ttl);


            /*!
             * \brief Ages the individuum
             *
             * \return `*this`
             */
            Individual &age();


            /*!
             * \brief Checks whether this individual is still alive or not
             *
             * \return true if the Individual's TTL is greater than 0, false
             *  otherwise.
             */
            bool isAlive() const;


            /*!
             * \brief Allows access to the error vector
             *
             * \return A modifiable reference to the error vector
             */
            ValueVector &errorVector();


            /*!
             * \brief Returns a read-only copy of the error vector
             *
             * \return The error vector write-protected
             */
            const ValueVector &errorVector() const;


            /*!
             * \brief Compares one individual to another.
             *
             * \param other The other individual
             *
             * \return -1 if the other individual is better than this one,
             *  0 if they are equal, or 1 if this one is better than the other
             *  individual.
             */
            int compare(const Individual &other) const;


            /*!
             * \brief Checks whether this object is better than another one.
             *
             * \param[in] other The object to check againts
             *
             * \return true if the object is better, false otherwise.
             */
            bool isBetterThan(const Individual &other) const;


            static bool isIndividual1Better(
                    const Individual *const &i1,
                    const Individual *const &i2);


            /*!
             * \brief Compares two Individuals for equality in all vectors.
             *
             * \param[in] other The Individual to compare the current one to.
             *
             * \return true iff equal, false otherwise.
             */
            bool operator==(const Individual &other) const;


            /*!
             * \brief Deep copy operator
             *
             * \param[in] other The other Individual to copy
             *
             * \return A deep copy
             */
            Individual &operator=(const Individual &other);
        };


        class REvolutionaryTrainingAlgorithm:
                public QObject,
                public TrainingAlgorithm
        {
            Q_OBJECT


        private:


            /*!
             * \brief Maximum number of epochs that may pass without a global
             *  improvement
             */
            size_t m_maxNoSuccessEpochs;


            /*!
             * \brief Overall size of the population
             */
            size_t m_populationSize;


            /*!
             * \brief Size of the elite, contained in the population
             */
            size_t m_eliteSize;


            /*!
             * \brief Weight of implicit gradient information.
             */
            double m_gradientWeight;


            /*!
             * \brief Weight of the reproduction success
             */
            double m_successWeight;


            /*!
             * \brief Smallest absolute delta; typically the smallest number
             *  we can store
             */
            double m_eamin;


            /*!
             * \brief Smallest relative delta
             */
            double m_ebmin;


            /*!
             * \brief The biggest relative change
             */
            double m_ebmax;


            /*!
             * \brief Initial Time-To-Live for new individuals
             */
            int m_startTTL;


            /*!
             * \brief Number of epochs to apply to the dc1 method
             */
            size_t m_measurementEpochs;


            /*!
             * \brief Success of reproduction
             */
            double m_success;


            /*!
             * \brief Target value on which the population has reached
             *  equilibrium
             */
            double m_targetSuccess;


            /*!
             * \brief Our Random Number Generator
             */
            boost::random::mt11213b m_randomNumberGenerator;


            /*!
             * \brief A uniform distribution `[0; 1)` from which we draw
             *  random numbers
             */
            boost::random::uniform_01<double> m_uniformDistribution;


            /*!
             * \brief Applies the bounds defined in ebmin, eamin and eamax
             *  given another object's parameter
             *
             * \param[in] dx The delta X that shall be checked and corrected
             *
             * \param[in] parameter Another object's parameter
             *
             * \return The corrected delta X
             */
            double applyDxBounds(const double &dx, const double &parameter)
                    const;


            /*!
             * \brief Checks that all parameters are within safe bounds
             *
             * \return true iff all parameters are in sensible bounds
             */
            bool hasSensibleTrainingParameters() const;


        public:


            /*!
             * \brief A time-discrete LTI system of first order
             *
             * \param[in] y
             * \param[in] u
             * \param[in] t
             *
             * \return
             */
            static double dc1(const double &y, const double &u, const double &t);


            /*!
             * \brief Sorts the population so that the best individual comes
             *  first.
             *
             * \param[inout] population The population.
             */
            static void sortPopulation(QList<Individual *> &population);


            /*!
             * \brief Creates a new instance of the
             *  evolutionary training algorithm for
             *  training a particular network.
             */
            REvolutionaryTrainingAlgorithm();


            /*!
             * \brief Creates a random number in the interval [0.0, 1.0)
             *
             * \return A random number between 0.0 and 1.0 (exclusive)
             */
            double frandom();


            /*!
             * \brief Returns the maximum number of epochs that may pass without
             *  a global improvement before the algorithm stops
             *
             * \return The number of epochs
             */
            size_t maxNoSuccessEpochs() const;


            /*!
             * \brief Sets the maximum number of epochs that may pass without
             *  a global improvement
             *
             * \param[in] epochs The new maximum number of epochs
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &maxNoSuccessEpochs(
                    const size_t &epochs);


            /*!
             * \brief Returns the size of the population
             *
             * \return The population's size
             */
            size_t populationSize() const;


            /*!
             * \brief Sets the size of the population
             *
             * \param[in] size The new population size
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &populationSize(
                    const size_t &size);


            /*!
             * \brief Returns the size of the elite
             *
             * \return The number of elite individuals within the population
             */
            size_t eliteSize() const;


            /*!
             * \brief Sets the number of elite objects within the population
             *
             * \param[in] size The new size of the elite
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &eliteSize(const size_t &size);


            /*!
             * \brief Returns the currently set gradient weight
             *
             * \return The gradient weight
             */
            double gradientWeight() const;


            /*!
             * \brief Sets the new gradient weight
             *
             * This weight factor sets the influence of the gradient information
             * on the training process. Setting it to 0.0 completeley disables
             * this feature. Values between [1.0, 3.0] typically yield the best
             * results. Values > 5.0 are probably not useful.
             *
             * \param weight The new weight
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &gradientWeight(
                    const double &weight);


            /*!
             * \brief Retrieves the weight factor applied to the error metric
             *
             * \return The error weight
             */
            double successWeight() const;


            /*!
             * \brief Sets the new error weight
             *
             * This value influences how much a given reproduction success
             * is weighted in when creating a new offspring and its parameters.
             *
             * \param weight The new weight
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &successWeight(
                    const double &weight);


            /*!
             * \brief The smallest absolute change applied during object
             *  creation
             *
             * \return The smallest absolute delta
             */
            double eamin() const;


            /*!
             * \brief Sets the smallest absolute delta for parameters
             *  or scatter
             *
             * This is the smallest absolute change we apply; smaller values
             * are not accepted. Typically, this is the smallest floating
             * point number the architecture accepts and acts as a safety net.
             *
             * \param eamin The new smallest absolute delta
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &eamin(const double &eamin);


            /*!
             * \brief Retrieves the current smallest relative change
             *
             * \return The smallest relative change for scatter/parameters
             */
            double ebmin() const;


            /*!
             * \brief Sets the new smallest relative delta applied to scatter
             *  and parameters
             *
             * We default to an `ebmin` so that `|(1.0 + ebmin) > 1.0|`.
             *
             * \param ebmin The new relative minimum
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &ebmin(const double &ebmin);


            /*!
             * \brief The relative maximum delta for scatter and parameters
             *
             * \return The relative maximum
             */
            double ebmax() const;


            /*!
             * \brief Sets the new relative maximum delta for scatter and
             *  parameter changes
             *
             * We default to an ebmax that does not lead to a too big spread
             * of individuals through reproduction: `ebmax < 10.0`.
             *
             * \param ebmax The new maximum delta
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &ebmax(const double &ebmax);


            /*!
             * \brief The default start TTL for new Individuals
             *
             * This variable is used when generating new objects. It should
             * be set before starting the training. A fail-safe default is
             * applied if not set: `ceil(0.1 * maxEpochs)`.
             *
             * \return The initial Time-To-Live
             */
            int startTTL() const;


            /*!
             * \brief Sets the new initial TTL for new Individuals
             *
             * \param[in] ttl The new TTL
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &startTTL(const int &ttl);


            /*!
             * \brief Mean number of epochs for application in the dc1 method.
             *
             * \return Number of epochs
             */
            size_t measurementEpochs() const;


            /*!
             * \brief Number of epochs applied to the dc1 method
             *
             * \param[in] epochs The number of epochs
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &measurementEpochs(
                    const size_t &epochs);


            /*!
             * \brief Generates the initial population from the supplied base
             *  network
             *
             * \param[in] baseNetwork The base network the user supplied for
             *  training
             *
             * \return The population, including the elite
             */
            QList<Individual *> generateInitialPopulation(
                    const NeuralNetwork &baseNetwork);


            /*!
             * \brief Creates a new neural network as part of the combination
             *  and crossover process
             *
             * This method generates a new offspring ANN by crossover and
             * mutation. It selects an ANN of the elite and one normal object.
             *
             * In the process, it also modifies existing objects in order to
             * re-train them.
             *
             * \param population The current population of networks.
             *
             * \return A newly generated ANN
             */
            Individual *modifyIndividual(
                    Individual *const &individual,
                    QList<Individual *> &population);



            /*!
             * \brief Trains the Neural Network using Ruppert's evolutionary
             *  training algorithm.
             *
             * \param trainingSet
             */
            virtual void train(
                    NeuralNetwork &ann,
                    TrainingSet &trainingSet)
                    override;

        signals:


            /*!
             * \brief Fired whenever an iteration is complete
             *
             * \param epoch The current epoch (i.e., nth iteration)
             *
             * \param error The current error
             *
             * \param population The complete population
             */
            void iterationFinished(
                    const size_t &epoch,
                    const double &error,
                    const QList<Individual *> &population);

        };
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::Individual &individual);
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::REvolutionaryTrainingAlgorithm &algorithm);
    ostream &operator<<(
            ostream &os,
            const QList<const Winzent::ANN::Individual *> &population);
}


#endif // WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
