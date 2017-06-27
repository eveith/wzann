#ifndef WINZENT_ANN_RPROPTRAININGALGORITHM_H
#define WINZENT_ANN_RPROPTRAININGALGORITHM_H


#include <unordered_map>

#include "TrainingSet.h"
#include "BackpropagationTrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class TrainingSet;
        class NeuralNetwork;


        /*!
         * \brief Trains a neural network using the iRPROP+ algorithm.
         *
         * iRPROP+ is an improved version of the original Resilient
         * BackPropagation training algorithm. It uses weight backtracking which
         * reverts the weight change of the last iteration if the sign of the
         * gradient changes in the current iteration.
         *
         * Some research suggests that iRPROP+ is the optimum RPROP algorithm.
         */
        class RpropTrainingAlgorithm : public TrainingAlgorithm
        {
        public:


            typedef std::unordered_map<
                    Connection*,
                    double> ConnectionGradientMap;


            /*!
             * \brief Positive step value
             */
            const static double ETA_POSITIVE;


            /*!
             * \brief Negative step value
             */
            const static double ETA_NEGATIVE;


            /*!
             * \brief Tolerance within which a value is still considered to be
             *  equal to 0.
             */
            const static double ZERO_TOLERANCE;


            /*!
             * \brief The initial value for weight changes
             */
            const static double DEFAULT_INITIAL_UPDATE;


            /*!
             * \brief The minimum delta value applied during weight change
             */
            const static double DELTA_MIN;


            /*!
             * \brief Maximum value for a delta during weight change
             */
            const static double MAX_STEP;


            /*!
             * \brief Returns the sign of a number, taking the zero tolerance
             *  into account
             *
             * \param[in] x The number we want to retrieve the sign of
             *
             * \return -1 on negative sign, 0 on 0, +1 on positive sign
             */
            static int sgn(double x);


            /*!
             * \brief Creates a new training algorithm instance for the given
             *  neural network.
             */
            RpropTrainingAlgorithm();


            /*!
             * \brief Trains the neural network
             *
             * \param[in] trainingSet A set of training data
             */
            virtual void train(NeuralNetwork& ann, TrainingSet& trainingSet)
                    override;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_RPROPTRAININGALGORITHM_H
