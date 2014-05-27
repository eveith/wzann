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
             * \brief Tolerance within which a value is still considered to be
             *  equal to 0.
             */
            const static qreal ZERO_TOLERANCE;


            /*!
             * \brief Returns the sign of a number, taking the zero tolerance
             *  into account
             *
             * \param[in] x The number we want to retrieve the sign of
             *
             * \return -1 on negative sign, 0 on 0, +1 on positive sign
             */
            static int sgn(const qreal &x);


            /*!
             * \brief Calculates the error at the output layer
             *
             * \param[in] expected The output that was expected from the network
             *
             * \param[in] actual The actual output the network emitted
             *
             * \return The error vector
             */
            ValueVector outputError(
                    const ValueVector &expected,
                    const ValueVector &actual)
                        const;



            /*!
             * Calculates the neuron delta for a neuron in an hidden layer.
             *
             * The delta is calculated by applying the derivative of the
             * neuron's activation function. So the neuron's activation function
             * must be differentiable.
             *
             * \param[in] neuron The hidden layer neuron
             *
             * \param[in] neuronDeltas A memoization hash for the deltas of
             *  other neurons.
             *
             * \param[in] outputError The definitive error values at the output
             *  layer
             *
             * \see #outputError
             */
            qreal hiddenNeuronDelta(
                    Neuron *const &neuron,
                    QHash<Neuron *, qreal> &neuronDeltas,
                    const ValueVector &outputError)
                        const;


            /*!
             * \brief Calculates the delta of a Neuron in the output layer
             *
             * The delta is calculated by applying the derivative of the
             * neuron's activation function. So the neuron's activation function
             * must be differentiable.
             *
             * \param[in] neuron The output layer neuron
             *
             * \param[in] error The neuron's error, i.e. the difference between
             *  the expected output value of that particular neuron and the
             *  actual value: $error = expected - actual$
             *
             * \return The delta
             */
            qreal outputNeuronDelta(
                    const Neuron *const &neuron,
                    const qreal &error) const;


            /*!
             * \brief Transparently computes the delta of a neuron
             *
             * \param[in] neuron A hidden or output layer neuron
             *
             * \param[inout] neuronDeltas Memoization hash for storing
             *  computations.
             *
             * \param[in] outputError The error at the output layer
             *
             * \return The neuron's delta
             *
             * \see #outputError
             */
            qreal neuronDelta(
                    const Neuron *const &neuron,
                    QHash<Neuron *, qreal> &neuronDeltas,
                    const ValueVector &outputError)
                        const;


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
