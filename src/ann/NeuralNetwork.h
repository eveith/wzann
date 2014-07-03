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
#include <QHash>

#include <functional>

#include <log4cxx/logger.h>

#include "Exception.h"


class QTextStream;


namespace Winzent {
    namespace ANN {
        typedef QVector<qreal> ValueVector;

        class NeuralNetworkPattern;
        class Layer;
        class Neuron;
        class Connection;

        class TrainingSet;
        class TrainingAlgorithm;


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
         * good output. Training is done using training strategies;
         * A network can be training using different strategies. Such classes
         * are derived from the ::TrainingAlgorithm class, e.g. the
         * ::BackpropagationTrainingAlgorithm or the
         * ::SimulatedAnnealingTrainingAlgorithm. Have a look at them for
         * training.
         *
         * \sa NeuralNetworkPattern
         *
         * \sa AbstractTrainingStrategy
         *
         * \sa Layer
         *
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
             * \brief The bias neuron, connected to each neuron in each layer.
             */
            Neuron *m_biasNeuron;


            /*!
             * All neurons that make up this neural network.
             */
            QList<Layer *> m_layers;


            /*!
             * An hash that indixes all connection originating from a certain
             * neuron.
             */
            QHash<Neuron*, QList<Connection*> > m_connectionSources;


            /*!
             * An hash that indexes all connections that lead to a certain
             * neuron.
             */
            QHash<Neuron*, QList<Connection*> > m_connectionDestinations;


            /*!
             * The pattern that is used to construct the network
             * and calculate a run through it.
             *
             * Patterns do two things: First, they define the layout
             * of a network, and second, they define how a network
             * works when values are calculated with it.
             */
            NeuralNetworkPattern *m_pattern;


        protected:


            /*!
             * \brief Internal logger
             */
            static log4cxx::LoggerPtr logger;


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
             * \return A clone of this current network.
             */
            NeuralNetwork *clone() const;


            /*!
             * \brief Returns this network's bias neuron
             *
             * \return The bias neuron of this Neural Network
             */
            const Neuron *biasNeuron() const;


            /*!
             * \brief Returns a modifiable version of the network's bias neuron
             *
             * \return The bias neuron, modifiable
             */
            Neuron *const &biasNeuron();


            /*!
             * \brief Sets a new bias neuron
             *
             * \param[in] neuron The new bias neuron
             *
             * \return `*this`
             */
            NeuralNetwork &biasNeuron(Neuron *neuron);


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
             * Checks whether two neurons <code>i</code> and
             * <code>j</code> are connected or not. Please note that
             * this method is not transitive, i.e.
             * <code>neuronConnectionExists(i, j)</code> is something
             * different than
             * <code>neuronConnectionExists(j, i)</code>.
             *
             * \param[in] from The "from" neuron
             * \param[in] to The "to" neuron
             *
             * \return <code>true</code> if a connection exists, false
             *  otherwise.
             */
            bool neuronConnectionExists(const Neuron *from, const Neuron *to)
                    const;


            /*!
             * Connects two neurons. Does not set a weight.
             *
             * \param from The neuron the connection originates
             *
             * \param to The destination neuron
             *
             * \return The new connection
             */
            Connection *connectNeurons(Neuron *const &from, Neuron *const &to)
                    throw(UnknownNeuronException);


            /*!
             * Retrieves the connection between two neurons.
             * If no weight exists, an exception is thrown.
             *
             * \param[in] form The neuron from which the connection
             *  begins
             *
             * \param[in] to The neuron to which the connection leads
             *
             * \return The connection, or NULL if no such connection exists
             *  (in which case an exception is thrown anyways).
             *
             * \throw NoConnectionException
             */
            Connection *neuronConnection(
                    const Neuron *const &from,
                    const Neuron *const &to)
                        const
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
             *
             * \throws WeightFixedException If the weight has been flagged as
             *  fixed.
             *
             * \sa Connection#weight
             */
            void weight(
                    const Neuron *const &from,
                    const Neuron *const &to,
                    double value)
                        throw(NoConnectionException, WeightFixedException);


            /*!
             * Finds all neurons to which a particular neuron is connected.
             *
             * The neuron is considered as the source of the connection.
             *
             * \param[in] neuron The neuron from which to find all available
             *  connections.
             *
             * \return A list containing all connection where the specified
             *  neuron is the source.
             *
             * \sa #neuronConnectionsTo
             */
            const QList<Connection*> neuronConnectionsFrom(
                    const Neuron *const &neuron)
                        const
                        throw(UnknownNeuronException);


            /*!
             * Finds all neurons which are the source of connections to the
             * specified one.
             *
             * \param[in] neuron The destination neuron
             *
             * \return A list containing all neurons that feed the specified
             *  one.
             *
             * \sa #neuronConnectionsFrom
             */
            QList<Connection*> neuronConnectionsTo(
                    const Neuron *const &neuron)
                        const
                        throw(UnknownNeuronException);



            /*!
             * \brief Iterates over each layer, yielding to the supplied lambda
             *  for each layer.
             *
             * \param yield The lambda that gets called for each connection.
             */
            void eachLayer(std::function<void(const Layer *const &)> yield)
                    const;


            /*!
             * \brief Iterates over all layers.
             *
             * This is an overloaded method. It yields a modifiable reference
             * to the layer.
             *
             * \param yield The lambda that is called for each layer, allowing
             *  layer modification.
             */
            void eachLayer(std::function<void(Layer *const &)> yield);



            /*!
             * \brief Iterates over all neuron connections in this network
             *
             * \param yield The lambda that is called for each neuron connection
             */
            void eachConnection(std::function<void(Connection *const&)> yield);


            /*!
             * \brief Iterates over all neuron connections read-only
             *
             * \param[in] yield The iterator lambda called for each neuron
             */
            void eachConnection(
                    std::function<void(const Connection *const &)> yield) const;


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
             * Returns the layer at the designated index.
             */
            Layer *&layerAt(const int &index);


            /*!
             * <code>const</code> version of #layerAt
             */
            Layer *const &layerAt(const int &index) const;


            /*!
             * Returns the layer at the designated index.
             */
            Layer*& operator [](const int &index);


            /*!
             * <code>const</code> version of the index operator.
             */
            Layer *operator [](const int &i) const;


            /*!
             * Returns the input layer
             */
            Layer *const &inputLayer() const;


            /*!
             * Returns the output layer
             */
            Layer *const &outputLayer() const;


            /*!
             * Configures this neural network based on a pattern.
             *
             * \param pattern The pattern that is used to set this
             *  network up. We create our own clone of the pattern
             *  in the process since the pattern is also used for
             *  calculation.
             *
             * \return `*this`
             */
            NeuralNetwork &configure(const NeuralNetworkPattern &pattern);


            /*!
             * \deprecated
             */
            NeuralNetwork &configure(
                    const NeuralNetworkPattern *const &pattern);


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
            ValueVector calculateLayer(
                    Layer *const &layer,
                    const ValueVector &input)
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
                    const int &from,
                    const int &to,
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
        };


        /*!
         * Appends the JSON representation of this network to the
         * designated data stream.
         */
        QTextStream& operator<<(QTextStream &out, const NeuralNetwork &network);
    } /* namespace ANN */
} /* namespace Winzent */


namespace std {
    ostream &operator<<(
            ostream &os,
            const Winzent::ANN::ValueVector &valueVector);
}


#endif /* NEURALNETWORK_H_ */
