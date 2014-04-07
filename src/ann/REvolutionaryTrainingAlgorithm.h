/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de>
 * \date 2013-03-19
 */


#ifndef WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
#define WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H


#include <QObject>

#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
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
             * \brief The ANN represented by this individual
             */
            NeuralNetwork *m_neuralNetwork;


            /*!
             * \brief The scatter vector of this individual
             */
            QList<qreal> m_scatter;


            /*!
             * \brief How many iterations this individual may exist
             */
            int m_timeToLive;


            /*!
             * \brief Stores the result of the last runs. Index 0 stores the
             *  overall error, indices 1..n stores the results of the indivdual
             *  runs.
             */
            QList<qreal> m_errorVector;


        public:


            /*!
             * \brief Creates a new individual given a neural network
             *
             * \param neuralNetwork The underlying ANN
             */
            explicit Individual(NeuralNetwork *neuralNetwork);


            /*!
             * \brief Destructs the underlying ANN.
             *
             * The destructor also destroys the underlying ANN. However,
             * the #neuralNetwork() accessor creates a clone that is returned.
             */
            virtual ~Individual();


            /*!
             * \brief Retrives a clone of the underlying neural network
             *
             * \return The underlying neural network
             */
            NeuralNetwork *neuralNetwork() const;


            /*!
             * \brief Retrieves the current scatter vector
             *
             * \return The current scatter vector
             */
            QList<qreal> scatter() const;


            /*!
             * \brief Sets the new scatter vector
             *
             * \param scatter The new scatter vector
             *
             * \return `*this`
             */
            Individual &scatter(QList<qreal> scatter);


            /*!
             * \brief Retrieves the current parameters set.
             *
             * \return The current parameters
             */
            QList<qreal> parameters() const;


            /*!
             * \brief Sets new parameters to this individual
             *
             * Sets new parameters to the individual. It modifies the underlying
             * ANN, too.
             *
             * \param parameters The new set of parameters
             *
             * \return `*this`
             */
            Individual &parameters(QList<qreal> parameters);


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
             * \brief Allows access to the error vector
             *
             * \return A modifiable copy of the error vector
             */
            QList<qreal> &errorVector();


            /*!
             * \brief Returns a read-only copy of the error vector
             *
             * \return The error vector write-protected
             */
            const QList<qreal> &errorVector() const;


            /*!
             * \brief Compares one individual to another.
             *
             * \param other The other individual
             *
             * \return -1 if the other individual is better than this one,
             *  0 if they are equal, or 1 if this one is better than the other
             *  individual.
             */
            int compare(const Individual *other) const;


            /*!
             * \brief Checks whether this object is better than another one.
             *
             * \param[in] other The object to check againts
             *
             * \return true if the object is better, false otherwise.
             */
            bool isBetterThan(const Individual *other) const;
        };

        
        class REvolutionaryTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            /*!
             * \brief Overall size of the population
             */
            int m_populationSize;


            /*!
             * \brief Size of the elite, contained in the population
             */
            int m_eliteSize;


            /*!
             * \brief Weight of implicit gradient information.
             */
            qreal m_gradientWeight;


            /*!
             * \brief Weight of the error returned for ANN training
             */
            qreal m_errorWeight;


            /*!
             * \brief Smallest absolute delta; typically the smallest number we
             *  can safe
             */
            qreal m_eamin;


            /*!
             * \brief Smallest relative delta
             */
            qreal m_ebmin;


            /*!
             * \brief The biggest relative change
             */
            qreal m_ebmax;


            /*!
             * \brief Applies the bounds defined in ebmin, eamin and eamax given
             *  another object's parameter
             *
             * \param dx The delta X that shall be checked and corrected
             *
             * \param parameter Another object's parameter
             *
             * \return The corrected delta X
             */
            qreal applyDxBounds(const qreal &dx, const qreal &parameter) const;


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
            static qreal dc1(const qreal &y, const qreal &u, const qreal &t);


            /*!
             * Creates a new instance of the evolutionary training algorithm for
             * training a particular network.
             *
             * \param network The network that shall be trained
             *
             * \param parent The parent object; if `0`, the target network
             *  becomes the parent object.
             */
            explicit REvolutionaryTrainingAlgorithm(
                    NeuralNetwork *const &network,
                    QObject *parent = 0);


            /*!
             * \brief Creates a random number in the interval [0.0, 1.0)
             *
             * \return A random number between 0.0 and 1.0 (exclusive)
             */
            static qreal frandom();


            /*!
             * \brief Returns the size of the population
             *
             * \return The population's size
             */
            int populationSize() const;


            /*!
             * \brief Sets the size of the population
             *
             * \param[in] size The new population size
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &populationSize(const int &size);


            /*!
             * \brief Returns the size of the elite
             *
             * \return The number of elite individuals within the population
             */
            int eliteSize() const;


            /*!
             * \brief Sets the number of elite objects within the population
             *
             * \param[in] size The new size of the elite
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &eliteSize(const int &size);


            /*!
             * \brief Returns the currently set gradient weight
             *
             * \return The gradient weight
             */
            qreal gradientWeight() const;


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
            REvolutionaryTrainingAlgorithm &gradientWeight(const qreal &weight);


            /*!
             * \brief Retrieves the weight factor applied to the error metric
             *
             * \return The error weight
             */
            qreal errorWeight() const;


            /*!
             * \brief Sets the new error weight
             *
             * This value influences how much a given error is weighted in when
             * creating a new offspring and its parameters.
             *
             * \param weight The new weight
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &errorWeight(const qreal &weight);


            /*!
             * \brief The smallest absolute change applied during object
             *  creation
             *
             * \return The smallest absolute delta
             */
            qreal eamin() const;


            /*!
             * \brief Sets the smallest absolute delta for parameters or scatter
             *
             * This is the smallest absolute change we apply; smaller values
             * are not accepted. Typically, this is the smallest floating
             * point number the architecture accepts and acts as a safety net.
             *
             * \param eamin The new smallest absolute delta
             *
             * \return `*this`
             */
            REvolutionaryTrainingAlgorithm &eamin(const qreal &eamin);


            /*!
             * \brief Retrieves the current smallest relative change
             *
             * \return The smallest relative change for scatter/parameters
             */
            qreal ebmin() const;


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
            REvolutionaryTrainingAlgorithm &ebmin(const qreal &ebmin);


            /*!
             * \brief The relative maximum delta for scatter and parameters
             *
             * \return The relative maximum
             */
            qreal ebmax() const;


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
            REvolutionaryTrainingAlgorithm &ebmax(const qreal &ebmax);


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
            Individual *generateIndividual(
                    const QList<Individual *> &population,
                    TrainingSet *const &trainingSet);



            /*!
             * \brief Trains the Neural Network using Ruppert's evolutionary
             *  training algorithm.
             *
             * \param trainingSet
             */
            virtual void train(TrainingSet *const &trainingSet);

        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
