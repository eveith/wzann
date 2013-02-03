/*!
 * \file	NeuralNetwork.h
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_


#include <QObject>
#include <QList>
#include <QVector>

#include "Exception.h"


class QTextStream;


namespace Winzent
{
    namespace ANN
    {
        typedef QVector<double> ValueVector;

        class NeuralNetworkPattern;
        class Neuron;

        class TrainingSet;
        class TrainingAlgorithm;


        /*!
         * Represents the weight of a connection between two neurons,
         * together with its properties.
         */
        class Weight: public QObject
        {
            Q_OBJECT

        public:


            /*!
             * The weight value
             */
            double value;


            /*!
             * Whether the value is fixed or not: Fixed values cannot
             * be changed by training.
             */
            bool fixed;


            /*!
             * Standard constructor which initialies a variable weight
             * with a weight value of 1.0.
             *
             * \sa #value
             *
             * \sa #fixed
             */
            Weight(QObject *parent = 0);


            /*!
             * Returns a clone of the current weight object, which acts
             * as a prototype.
             */
            Weight* clone() const;


            /*!
             * Returns the current weight value.
             */
            double weight() const;


            /*!
             * Sets a new weight. Throws an instance of
             * WeightFixedException if the weight is fixed.
             *
             * \param weight The new weight
             *
             * \throw WeightFixedException If the weight is fixed.
             */
            void weight(double weight) throw(WeightFixedException);


            /*!
             * Sets the weight to a random value between
             * <code>min</code> and <code>max</code>.
             *
             * \input[in] min The minimum inclusive value
             *
             * \input[in] max The maximum inclusive value
             *
             * \throws WeightFixedException if the weight is fixed.
             */
            void setRandomWeight(const double &min, const double &max)
                    throw(WeightFixedException);


            /*!
             * Directly returns the weight value (conversion operator)
             */
            operator double() const;


            double operator*(const double &rhs) const;
        };


        /*!
         * Represents a layer within a neural network.
         */
        class Layer: public QObject
        {
            Q_OBJECT

        public:


            /*!
             * A list of all neurons the make up this layer.
             */
            QList<Neuron*> neurons;


            /*!
             * Returns the size of the layer, i.e. the number of
             * neurons it holds.
             */
            int size() const;


            /*!
             * Creates a new, empty layer.
             */
            Layer(QObject *parent = 0);


            /*!
             * Returns a deep copy (clone) of this layer.
             */
            Layer* clone() const;
        };


        /*!
         * A list of list of <code>double*</code>, used to store the
         * weights on connections between different neurons. If a
         * connection between two neurons does not exist, the pointer
         * is <code>NULL</code>. Otherwise, any double value is
         * allowed.
         *
         * Remember that a weight of 0 does not mean that
         * there is no connection, only that its weight is 0. While
         * this might seem to be the same, remember that a weight
         * of 0 can be modified to be of any other value, e.g. through
         * training, while a non-existent connection cannot be
         * intensified.
         */
        typedef QList<QList<Weight*> > WeightMatrix;


        /*!
         * Represents a Neural Network.
         *
         * Instances of this class represent any neural network. A
         * neural network is made up by layers which contain neurons.
         * Neurons are interconnected; each connection has a certain
         * weight. A neuron is basically nothing else than a
         * mathematical function, such as the sigmoid function. The
         * input of a neuron is the sum of all neurons which are
         * connected to it, modified by the weight of the connections
         * which lead to this one.
         *
         * One layer is the input layer; it contains the neurons
         * which interface with the world outside the network. Each
         * neuron in the input layer receives an input. This input is
         * designated by the parameter to the #calculate method. The
         * result is the output of the neurons in the output layer.
         *
         * Many layouts exist for neural networks. These layouts
         * are produced by NeuralNetworkPattern class instances.
         * Patterns are templates which define how the #calculate
         * method acts. A neural network is only complete with the
         * accompanying pattern class.
         *
         * Neural networks need to be trained in order to produce
         * good output. Training is done using training strategies.
         * A network can be training using different strategies.
         *
         * \sa NeuralNetworkPattern
         * \sa AbstractTrainingStrategy
         * \sa Layer
         * \sa Neuron
         */
        class NeuralNetwork: public QObject
        {
            Q_OBJECT


            friend class NeuralNetworkPattern;
            friend class AbstractTrainingStrategy;

            friend QTextStream& operator<<(
                    QTextStream &out,
                    const NeuralNetwork &network);


            private:


            /*!
             * All neurons that make up this neural network.
             */
            QList<Layer*> m_layers;


            /*!
             * Defines the weights between the
             */
            WeightMatrix m_weightMatrix;


            /*!
             * The pattern that is used to construct the network
             * and calculate a run through it.
             *
             * Patterns do two things: First, they define the layout
             * of a network, and second, they define how a network
             * works when values are calculated with it.
             */
            NeuralNetworkPattern *m_pattern;


        public:


            /*!
             * The library version, needed e.g. for serialization.
             */
            static const char VERSION[];


            /*!
             * Constructs an empty, uninitialized network.
             *
             * \param parent The parent object that owns this one
             */
            NeuralNetwork(QObject *parent = 0);


            /*!
             * Constructs a new neural network which gets initialized
             * with the layers supplied by the first argument.
             *
             * \param layers A list of layers (possibly with neurons)
             *  that make up this neural network
             *
             * \param parent The parent object that owns this one
             */
            NeuralNetwork(QList<Layer*> *layers, QObject *parent = 0);


            /*!
             * Copy constructor
             */
            NeuralNetwork(const NeuralNetwork &rhs);


            /*!
             * Destructs the neural network and calls
             * <code>delete</code> on all layers and neurons.
             */
            virtual ~NeuralNetwork();


            /*!
             * Checks whether a certain neuron is part of this
             * network or not.
             *
             * \param neuron The neuron to look for
             *
             * \return <code>true</code> if it lives in this network,
             *  <code>false</code> otherwise.
             */
            bool containsNeuron(const Neuron *neuron) const;


            /*!
             * Find a neuron in the 2D weight matrix and returns its
             * index. If the neuron is not part of this network, the
             * return value is undefined.
             *
             * \param neuron Pointer to the neuron
             *
             * \return The index of the neuron in this network's
             *  weight matrix
             *
             * \sa #containsNeuron
             */
            int findNeuron(const Neuron *neuron) const;


            /*!
             * Translates a neuron index identified by its layer's
             * index and the neuron's index within the layer to the
             * neuron's absolute index in the network's 2D weight
             * matrix.
             *
             * If the layer index is out of bounds, <code>-1</code>
             * is returned. If the neuron index is out of bounds,
             * the result is undefined.
             *
             * \param layer The layer in which the neuron resides,
             *  starting at index 0
             *
             * \param neuronIndex The index of the neuron within the
             *  layer identified by <code>layer</code>, also starting
             *  at index 0.
             *
             * \return The neuron's absolute index within the 2D
             *  weight matrix of this network
             */
            int translateIndex(const int &layer, const int &neuronIndex) const;


            /*!
             * Checks whether two neurons <code>i</code> and
             * <code>j</code> are connected or not. Please note that
             * this method is not transitive, i.e.
             * <code>neuronConnectionExists(i, j)</code> is something
             * different than
             * <code>neuronConnectionExists(j, i)</code>.
             *
             * \param[in] i The "from" neuron
             * \param[in] j The "to" neuron
             *
             * \return <code>true</code> if a connection exists, false
             *  otherwise.
             */
            bool neuronConnectionExists(const int &i, const int &j) const;


            /*!
             * Connects two neurons
             *
             * \param i The index of the originating neuron
             *  in the 2D weight matrix
             *
             * \param j The index of the destination neuron in the
             *  2D weight matrix
             */
            void connectNeurons(const int &i, const int &j);


            /*!
             * Connects two neurons with a random weight.
             *
             * \param i The neuron the connection originates
             *
             * \param j The destination neuron
             */
            void connectNeurons(Neuron *i, Neuron *j);


            /*!
             * Retrieves the weight on the connection between two
             * neurons. If no weight exists, an exception is thrown.
             *
             * \param[in] i The neuron from which the connection
             *  begins
             * \param[in] j The neuron to which the connection leads
             *
             * \return The weight as double
             *
             * \throw NoConnectionException
             */
            Weight* weight(const int &i, const int &j) const
                    throw(NoConnectionException);


            /*!
             * Sets a new weight on a connection
             *
             * \param[in] i The neuron from which the connection
             *  originates
             *
             * \param[in] j The neuron that is the destination of the
             *  connection
             *
             * \param[in] value The new weight value
             *
             * \throws NoConnectionException If the connection does
             *  not exist.
             */
            void weight(const int &i, const int &j, double value)
                    throw(NoConnectionException);


            /*!
             * Finds a neuron by its absolute index in this network's
             * 2D weight matrix. If the neuron does not exist,
             * <code>NULL</code> is returned instead.
             *
             * \param index The absolute index of the neuron
             *
             * \return The neuron if it has been found, or
             *  <code>NULL</code> otherwise.
             *
             * \sa #translateIndex
             */
            Neuron* neuron(const int &index);


            /*!
             * Adds a layer to the neural network. The layer is
             * appended to the end of the network and becomes the new
             * output layer. If it is the first layer, it also becomes
             * the input layer. If you do not want to follow this
             * logic, you can always explicitly set the input and
             * output layer.
             *
             * \sa #inputLayer
             * \sa #outputLayer
             */
            NeuralNetwork& operator<<(Layer *layer);


            /*!
             * Returns the size of the neural network in terms of
             * the number of layers. Useful in combination with the
             * index operator.
             *
             * \return The number of layers this network contains
             *
             * \sa #operator[]
             */
            int size() const
            {
                return m_layers.size();
            }


            /*!
             * Returns the operator at the designated index.
             */
            Layer*& operator[](const int &i);


            /*!
             * Explicitly set the input layer
             */
            void inputLayer(Layer *layer);


            /*!
             * Returns the input layer
             */
            Layer* inputLayer() const;


            /*!
             * Explicitly sets the output layer
             */
            void outputLayer(Layer *layer);


            /*!
             * Returns the output layer
             */
            Layer* outputLayer() const;


            /*!
             * Configures this neural network based on a pattern.
             *
             * \param pattern The pattern that is used to set this
             *  network up. We create our own clone of the pattern
             *  in the process since the pattern is also used for
             *  calculation.
             */
            void configure(const NeuralNetworkPattern *pattern);


            /*!
             * Feeds each neuron on a given layer and returns the
             * complete output. The number of values in the
             * <code>input</code> vector must match the number of
             * neurons on the layer, otherwise an exception is thrown.
             *
             * \param[in] layer The layer whose neurons are to be
             *  fed
             * \param[in] input The input for each neuron
             *
             * \return  The output of each neuron
             *
             * \throws LayerSizeMismatchException If the size of the
             *  input vector does not match the number of neurons in
             *  the layer
             *
             * \sa Neuron#activate
             *
             * \sa #calculateLayer
             */
            ValueVector calculateLayer(Layer *layer, const ValueVector &input)
                    throw(LayerSizeMismatchException);


            /*!
             * A shortcut for
             *
             *      calculateLayer(*(network)[layerIndex], input)
             *
             * \param layerIndex The index of the layer
             *
             * \param input The input to the layer's neurons
             *
             * \sa #calculateLayer
             *
             * \sa Neuron#activate
             *
             * \throw LayerSizeMismatchException
             */
            ValueVector calculateLayer(
                    const int &layerIndex,
                    const ValueVector &input)
                        throw(LayerSizeMismatchException);


            /*!
             * Calculates the transition of values from one layer
             * to another.
             *
             * The neurons of one layer are connected to neurons in
             * another layer. These two layers are identified by
             * their index.
             *
             * It is the responsibility of the caller to ensure that
             * a connection between the two layers actually exist.
             * Otherwise, due to technical reasons, the result
             * vector contains 0.0 for each neuron.
             *
             * \param fromLayer The index of the originating layer
             *
             * \param toLayer The index of the destination layer
             *
             * \param input The input that originates from the first
             *  layer and makes its transition to the target layer
             *
             * \return A value vector that constitutes the input of
             *  the target layer neurons. This vector always has the
             *  same size as the target layer.
             *
             * \sa #operator[]
             *
             * \throw LayerSizeMismatchException
             */
            ValueVector calculateLayerTransition(
                    const int &fromLayer,
                    const int &toLayer,
                    const ValueVector &input)
                        throw(LayerSizeMismatchException);


            /*!
             * Calculates a complete pass of the neural network.
             *
             * \input[in] input The input to the neural network,
             *  which is a vector that maps 1:1 an input value to an
             *  input neuron. If the size of the vector does not match
             *  the number of input neurons, an exception is thrown.
             */
            ValueVector calculate(const ValueVector &input)
                    throw(LayerSizeMismatchException);


            /*!
             * Trains the network using a certain strategy. This may,
             * of course, occur several times with the same net; the
             * new training uses the net as it is.
             *
             * Most training strategies operate on a set of training
             * data, which is supplied via the
             * <code>TrainingSet</code> container. This does not only
             * store the training patterns, but also the error and
             * defines the stopping point, either when a maximum
             * number of iterations is reach or the training is
             * successful because the training error drops below the
             * designated error value.
             *
             * This is basically a shortcut for calling
             * the <code>train()</code> method on a training strategy.
             *
             * \param trainigStrategy The training strategy to
             *  employ.
             *
             * \param trainingSet The training data to operate on
             *
             * \sa AbstractTrainingStrategy
             * \sa TrainingSet
             */
            void train(
                    TrainingAlgorithm *trainingStrategy,
                    TrainingSet *trainingSet);

        };


        /*!
         * Appends the JSON representation of this network to the
         * designated data stream.
         */
        QTextStream& operator<<(QTextStream &out, const NeuralNetwork &network);


    } /* namespace ANN */
} /* namespace Winzent */


#endif /* NEURALNETWORK_H_ */
