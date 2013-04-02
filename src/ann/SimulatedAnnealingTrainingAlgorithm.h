/*!
 * \file
 * \author Eric MSP Veith <eveith@veith-m.de>
 * \date 2013-03-21
 */


#ifndef SIMULATEDANNEALINGTRAININGALGORITHM_H
#define SIMULATEDANNEALINGTRAININGALGORITHM_H


#include <QObject>

#include "TrainingAlgorithm.h"


namespace Winzent {
    namespace ANN {


        class NeuralNetwork;
        class TrainingSet;


        class SimulatedAnnealingTrainingAlgorithm: public TrainingAlgorithm
        {
            Q_OBJECT

        private:


            /*!
             * Cutoff value for random data.
             */
            static const qreal CUT;


            /*!
             * The temperature at which the simulated annealing process starts.
             */
            qreal m_startTemperature;


            /*!
             * The temperature at which the simulated annealing process stops.
             */
            qreal m_stopTemperature;


            /*!
             * The number of cycles to run at a certain temperature.
             */
            int m_cycles;


            /*!
             * Randomizes the weights of a given neural network according to the
             * current state/temperature of the algorithm.
             *
             * \param[inout] network The neural network to randomize.
             */
            void randomize(NeuralNetwork *network, const qreal &temperature);


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
            qreal iterate(
                    NeuralNetwork *&network,
                    TrainingSet *trainingSet);


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
                    NeuralNetwork *const &network,
                    qreal startTemperature,
                    qreal stopTemperature,
                    int cycles,
                    QObject *parent = 0);


            /*!
             * \return The starting temperature
             */
            qreal startTemperature() const;


            /*!
             * \return The stopping temperature
             */
            qreal stopTemperature() const;


            /*!
             * \return The current temperature
             */
            qreal temperature() const;


            /*!
             * \return The number of cycles per temperature
             */
            int cycles() const;


            /*!
             * Trains the neural network using simulated annealing.
             *
             * \param trainingSet A set of training data
             */
            virtual void train(TrainingSet *const &trainingSet);
        };
    }
}

#endif // SIMULATEDANNEALINGTRAININGALGORITHM_H
