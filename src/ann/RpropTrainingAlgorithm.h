#ifndef WINZENT_ANN_RPROPTRAININGALGORITHM_H
#define WINZENT_ANN_RPROPTRAININGALGORITHM_H


#include <QObject>

#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class TrainingSet;


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
        class RpropTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        public:


            /*!
             * \brief Positive step value
             */
            const static qreal ETA_POSITIVE;


            /*!
             * \brief Negative step value
             */
            const static qreal ETA_NEGATIVE;


            /*!
             * \brief Creates a new training algorithm instance for the given
             *  neural network.
             *
             * \param[in] network The neural network to train
             *
             * \param parent The parent object
             */
            explicit RpropTrainingAlgorithm(
                    NeuralNetwork *const &network,
                    QObject *parent = 0);


            /*!
             * \brief Trains the neural network
             *
             * \param[in] trainingSet A set of training data
             */
            virtual void train(TrainingSet *const &trainingSet);
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_RPROPTRAININGALGORITHM_H
