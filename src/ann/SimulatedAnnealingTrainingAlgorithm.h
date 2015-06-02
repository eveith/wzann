/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de>
 * \date 2013-03-21
 */


#ifndef SIMULATEDANNEALINGTRAININGALGORITHM_H
#define SIMULATEDANNEALINGTRAININGALGORITHM_H


#include <QtGlobal>

#include <cstddef>

#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class TrainingSet;


        class SimulatedAnnealingTrainingAlgorithm: public TrainingAlgorithm
        {
        private:


            /*!
             * Cutoff value for random data.
             */
            static const double CUT;


            /*!
             * The temperature at which the simulated annealing process starts.
             */
            double m_startTemperature;


            /*!
             * The temperature at which the simulated annealing process stops.
             */
            double m_stopTemperature;


            /*!
             * The number of cycles to run at a certain temperature.
             */
            size_t m_cycles;


            /*!
             * \brief Extracts all weights as parameter vector from the given
             *  Neural Network
             *
             * \param[in] network The network to extract all trainable weights
             *  from
             *
             * \return The parameter vector
             */
            static ValueVector getParameters(const NeuralNetwork &network);


            /*!
             * \brief Applies the given parameters vector (i. e., weights) to
             *  the supplied neural network
             *
             * \param[in] parameters The vector of weights
             *
             * \param[inout] network The ANN to apply the parameters to
             */
            static void applyParameters(
                    const ValueVector &parameters,
                    NeuralNetwork &network);


            /*!
             * Randomizes the weights of a given neural network according to the
             * current state/temperature of the algorithm.
             *
             * \param[inout] network The neural network to randomize.
             */
            void randomize(ValueVector &parameters, const double &temperature);


            /*!
             * Do one iteration of #m_cycles cycles
             *
             * \param[inout] network The network that is to be trained. This
             *  network will be modified, so pass a copy if you do not want
             *  this.
             *
             * \param[in] trainingSet The set of training data
             *
             * \return The error at the end of the iteration
             */
            double iterate(
                    NeuralNetwork &network,
                    TrainingSet const &trainingSet);


        public:


            /*!
             * Constructs a new instance of this training algorithm
             *
             * \param[in] network The network to be trained
             *
             * \param[in] startTemperature The temperature value at which the
             *  training starts.
             *
             * \param[in] stopTemperature The temperature at which an epoch
             *  ends.
             *
             * \param[in] cycles The number of cycles needed to reach the
             *  stopTemperature
             */
            explicit SimulatedAnnealingTrainingAlgorithm(
                    double startTemperature,
                    double stopTemperature,
                    size_t cycles);


            /*!
             * \return The starting temperature
             */
            double startTemperature() const;


            /*!
             * \return The stopping temperature
             */
            double stopTemperature() const;


            /*!
             * \return The current temperature
             */
            double temperature() const;


            /*!
             * \return The number of cycles per temperature
             */
            size_t cycles() const;


            /*!
             * Trains the neural network using simulated annealing.
             *
             * \param trainingSet A set of training data
             */
            virtual void train(
                    NeuralNetwork &ann,
                    TrainingSet &trainingSet)
                    override;
        };
    }
}

#endif // SIMULATEDANNEALINGTRAININGALGORITHM_H
