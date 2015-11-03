#ifndef WINZENT_ANN_PSOTRAININGALGORITHM_H_
#define WINZENT_ANN_PSOTRAININGALGORITHM_H_


#include "ParticleSwarmOptimization.h"

#include "TrainingSet.h"
#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {


        /*!
         * \brief Trains an Artificial Neural Network using Particle Swarm
         *  Optimization
         *
         * \sa Winzent::Algorithm::ParticleSwarmOptimization
         */
        class WINZENTANNSHARED_EXPORT PsoTrainingAlgorithm:
                public TrainingAlgorithm,
                public Algorithm::ParticleSwarmOptimization
        {
        public:


            /*!
             * \brief Constructs a new instance with default parameters
             *  (mostly 0).
             */
            PsoTrainingAlgorithm();


            /*!
             * \brief Applies the particle's current position to a
             *  Artificial Neural Network
             *
             * \param[in] particle The particle whose position data should
             *  be applied
             *
             * \param[inout] neuralNetwork The ANN to which the particle's
             *  position data is applied
             *
             * \sa Algorithm::detail::Particle::currentPosition
             */
            static void applyPosition(
                    const Algorithm::detail::Particle &particle,
                    NeuralNetwork &neuralNetwork);


            /*!
             * \brief Evaluates a particle's fitness to solve the training
             *  set with the supplied ANN.
             *
             * \param[inout] particle The particle that carries the ANN's
             *  parameters
             *
             * \param[inout] ann The Artificial Neural Network to which we
             *  apply the particle's position in order to evaluate it
             *
             * \param[in] trainingSet The training set holding the data that
             *  this method uses to evaluate the particle
             */
            void evaluateParticle(
                    Algorithm::detail::Particle &particle,
                    NeuralNetwork &ann,
                    const TrainingSet &trainingSet);


            /*!
             * \brief Trains the given Artificial Neural Network (ANN) using
             *  Particle Swarm Optimization
             *
             * \param[in] ann The Artifical Neural Network to train
             *
             * \param[in] trainingSet The training data
             */
            virtual void train(NeuralNetwork &ann, TrainingSet &trainingSet)
                    override;


        private:


            //! The actual algorithm implementation
            Algorithm::ParticleSwarmOptimization m_Pso;
        };
    }
}


#endif
