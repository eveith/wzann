#ifndef WZANN_BACKPROPAGATIONTRAININGALGORITHM_H_
#define WZANN_BACKPROPAGATIONTRAININGALGORITHM_H_


#include <cmath>
#include <unordered_map>

#include "NeuralNetwork.h"
#include "TrainingAlgorithm.h"


namespace wzann {
    class Neuron;
    class TrainingSet;
    class NeuralNetwork;


    class BackpropagationTrainingAlgorithm : public TrainingAlgorithm
    {
    public:

        typedef std::unordered_map<
                Connection*,
                double> ConnectionDeltaMap;


        const double DEFAULT_LEARNING_RATE = 0.7;


        /*!
         * Constructs a new instance of the Backpropagation training
         * algorithm.
         *
         * \param[in] network The network to be trained
         *
         * \param[in] learningRate The learning rate applied to each weight
         *  change: Literature suggest `0.7` as a sensible starting value.
         *
         * \param parent The parent object
         */
        BackpropagationTrainingAlgorithm();


        /*!
         * \return The learning rate applied to each weight change
         */
        double learningRate() const;


        /*!
         * \brief Sets the learning rate of the Backpropagation training
         *  algorithm
         *
         * \param[in] rate The new rate
         *
         * \return `*this`
         */
        BackpropagationTrainingAlgorithm& learningRate(double rate);


        /*!
         * \brief Trains the neural network with the given trainingSet.
         *
         * \param[in] trainingSet A set of sample inputs and expected
         *  outputs.
         */
        virtual void train(NeuralNetwork& ann, TrainingSet& trainingSet)
                override;


    private:

        //! \brief The learning rate applied to each weight change
        double m_learningRate;
    };
} // namespace wzann

#endif // WZANN_BACKPROPAGATIONTRAININGALGORITHM_H_
