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

        
        class EvolutionaryTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            class Chromosome: public QObject
            {
            private:


                /*!
                 * A list with all "genes", i. e. the weights that make up this
                 * chromosome.
                 */
                QList<Connection *> m_genes;


                /*!
                 * The neural network which is the source of all "genes".
                 */
                NeuralNetwork *m_network;


                /*!
                 * The range by which a mutation is undertaken. For example, an
                 * amplitude of 4.0 will cause a mutation in the range of
                 * [-2.0, +2.0].
                 */
                qreal m_mutationAmplitude;


                /*!
                 * The default mutation amplitude used at initialization.
                 */
                static qreal defaultMutationAmplitude;


            public:


                /*!
                 * Creates a new chromosome from a given neural network.
                 *
                 * \param network The neural network which is the initial source
                 *  of all weights (i. e., genes).
                 */
                Chromosome(const NeuralNetwork *&network, QObject *parent = 0);


                /*!
                 * Mutates random chromosomes with a certain propability.
                 *
                 * \param[in] probability The propability for mutation. If the
                 *  propability is `1.0`, every gene will be mutated.
                 */
                void mutate(const double &probability);


                /*!
                 * Does a two-point crossover of weights but only within the
                 * layer connection; i. e. this version will only exchange
                 * connections whose source and target layer are the same.
                 *
                 * \param[in] network The neural network used for the crossover
                 *
                 * \param[in] probability The probability with which a crossover
                 *  happens.
                 */
                void crossoverIntraLayer(
                        const NeuralNetwork *&network,
                        const double &probability);
                /*
                void crossoverInterLayer(const double &propability);
                */


                /*!
                 * \return The current mutation amplitude
                 */
                qreal mutationAmplitude() const;


                /*!
                 * Changes the mutation amplitude
                 *
                 * \return `this`
                 */
                Chromosome *mutationAmplitude(qreal amplitude);
            };


            class Population: public QObject
            {
            public:

                QList<Chromosome *> members;
            };


        protected:


            /*!
             * Trains the neural network using an evolutionary approach.
             *
             * \param network The neural network that is to be trained
             *
             * \param trainingSet A couple of information about the training
             *  process, along with the training input.
             *
             * \sa TrainingSet
             */
            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet);


            /*!
             * Mutates a double value.
             *
             * \param[in] d The double that is to be mutated
             *
             * \return The mutated double
             */
            qreal mutate(const qreal &d) const;


            explicit EvolutionaryTrainingAlgorithm(QObject *parent = 0);
        };
        
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_EVOLUTIONARYTRAININGALGORITHM_H
