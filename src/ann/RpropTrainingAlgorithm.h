#ifndef WINZENT_ANN_RPROPTRAININGALGORITHM_H
#define WINZENT_ANN_RPROPTRAININGALGORITHM_H


#include "TrainingSet.h"
#include "TrainingAlgorithm.h"


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
        class RpropTrainingAlgorithm: public TrainingAlgorithm
        {
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
             * \brief The initial value for weight changes
             */
            const static qreal DEFAULT_INITIAL_UPDATE;


            /*!
             * \brief The minimum delta value applied during weight change
             */
            const static qreal DELTA_MIN;


            /*!
             * \brief Maximum value for a delta during weight change
             */
            const static qreal MAX_STEP;


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
             * \brief Runs a sample through the network and calculates the error
             *  at the output layer.
             *
             * \param[in] network The neural network
             *
             * \param[in] trainingItem The sample that is fed to the network
             *
             * \return A vector of errors at the output layer.
             */
            ValueVector feedForward(
                    NeuralNetwork &network,
                    const TrainingItem &trainingItem)
                    const;


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
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, qreal> &neuronDeltas,
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
                    const Neuron &neuron,
                    const qreal &error)
                    const;


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
                    NeuralNetwork &ann,
                    const Neuron &neuron,
                    QHash<const Neuron *, qreal> &neuronDeltas,
                    const ValueVector &outputError)
                    const;


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
            virtual void train(NeuralNetwork &ann, TrainingSet &trainingSet)
                    override;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_RPROPTRAININGALGORITHM_H
