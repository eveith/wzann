/*!
 * \file	AbstractTrainingStrategy.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef ABSTRACTTRAININGSTRATEGY_H_
#define ABSTRACTTRAININGSTRATEGY_H_

#include <qobject.h>

namespace Winzent
{
    namespace ANN
    {
        class NeuralNetwork;
        class TrainingSet;


        /*!
         * Derivatives of this class implement training strategies,
         * both supervised and unsupervised, which are used to
         * train neural networks.
         *
         * \sa NeuralNetwork#train
         */
        class AbstractTrainingStrategy: public QObject
        {
            Q_OBJECT

        public:


            AbstractTrainingStrategy(QObject *parent = 0);
            virtual ~AbstractTrainingStrategy();


            /*!
             * Trains a neural network with the concrete training
             * strategy, i.e. one that implements this method. It does
             * this using a certain set of training data.
             *
             * \param network The network that is to be trained
             *
             * \param trainingSet The training set class that holds
             *  the training data.
             */
            virtual void train(
                    NeuralNetwork *network,
                    TrainingSet *trainingSet) = 0;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* ABSTRACTTRAININGSTRATEGY_H_ */
