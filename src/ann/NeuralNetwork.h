/*!
 * \file	NeuralNetwork.h
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_


#include <vector>
#include <memory>
#include <cstddef>
#include <unordered_map>

#include <boost/range.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <log4cxx/logger.h>

#include <JsonSerializable.h>

#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Winzent-ANN_global.h"


class QTextStream;


using std::function;
using boost::ptr_vector;
using boost::make_iterator_range;


namespace Winzent {
    namespace ANN {
        class Connection;
        class NeuralNetworkPattern;

        class TrainingSet;
        class TrainingAlgorithm;


        /*!
         * \brief Represents a Neural Network.
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
        class WINZENTANNSHARED_EXPORT NeuralNetwork: public JsonSerializable
        {
            friend class NeuralNetworkPattern;
            friend class AbstractTrainingStrategy;


        public:


            typedef std::size_t size_type;
            typedef ptr_vector<Layer>::iterator LayerIterator;
            typedef ptr_vector<Layer>::const_iterator LayerConstIterator;

            typedef std::vector<Connection *> ConnectionsVector;
            typedef ConnectionsVector::iterator ConnectionIterator;
            typedef ConnectionsVector::const_iterator ConnectionConstIterator;
            typedef std::unordered_map<Neuron *, ConnectionsVector>
                    NeuronConnectionsMap;
            typedef std::pair<ConnectionIterator, ConnectionIterator>
                    ConnectionRange;
            typedef std::pair<
                        ConnectionConstIterator,
                        ConnectionConstIterator> ConnectionConstRange;

            //! The library version, needed e.g. for serialization.
            static const char VERSION[];


            //! Constructs an empty, uninitialized network.
            NeuralNetwork();


            //! Copy constructor
            NeuralNetwork(const NeuralNetwork &rhs);


            //! Deletes all neurons, connections, and activation functions
            virtual ~NeuralNetwork();


            /*!
             * \brief Contstructs a deep copy of the network
             *
             * \return A clone of this current network.
             */
            NeuralNetwork *clone() const;


            /*!
             * \brief Returns this network's bias neuron
             *
             * \return The bias neuron of this Neural Network
             */
            const Neuron &biasNeuron() const;


            /*!
             * \brief Returns a modifiable version of the network's bias neuron
             *
             * \return The bias neuron, modifiable
             */
            Neuron &biasNeuron();


            /*!
             * Checks whether a certain neuron is part of this
             * network or not.
             *
             * \param neuron The neuron to look for
             *
             * \return <code>true</code> if it lives in this network,
             *  <code>false</code> otherwise.
             */
            bool contains(const Neuron &neuron) const;


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
            bool connectionExists(const Neuron &from, const Neuron &to) const;


            /*!
             * Connects two neurons. Does not set a weight.
             *
             * \param from The neuron the connection originates
             *
             * \param to The destination neuron
             *
             * \return The new connection
             */
            Connection &connectNeurons(const Neuron &from, const Neuron &to);


            /*!
             * \brief Removes the connection between two neurons
             *
             * \param[in] from The source neuron
             *
             * \param[in] to The destination neuron
             */
            void disconnectNeurons(const Neuron &from, const Neuron &to);


            /*!
             * \brief Retrieves the connection between two neurons.
             *
             * This returns the connection between two neurons. If no such
             * connection exists, an exception is thrown.
             *
             * \param[in] form The neuron from which the connection
             *  begins
             *
             * \param[in] to The neuron to which the connection leads
             *
             * \return A pointer to the Connection in question, or a
             *  `nullptr` if the two neurons are not connected.
             *  Then, an exception is also thrown.
             *
             * \throw NoConnectionException
             */
            Connection *connection(const Neuron &from, const Neuron &to);


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
            ConnectionRange connectionsFrom(const Neuron &neuron);


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
            ConnectionConstRange connectionsFrom(const Neuron &neuron) const;

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
            ConnectionRange connectionsTo(const Neuron &neuron);


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
            ConnectionConstRange connectionsTo(const Neuron &neuron) const;


            /*!
             * \brief Allows to iterate over all layers in the ANN
             *
             * \return A pair of layer iterators: `[begin, end)`
             */
            std::pair<LayerIterator, LayerIterator> layers();


            /*!
             * \brief Allows to iterate over read-only-accessible layers
             *
             * \return A pair of const iterators over all layers
             */
            std::pair<LayerConstIterator, LayerConstIterator> layers() const;


            /*!
             * \brief Iterates over all neuron connections read-only
             *
             * \param[in] yield The iterator lambda called for each neuron
             */
            template<class UnaryFunction>
            void eachConnection(UnaryFunction f)
            {
                for (const auto &layer: make_iterator_range(layers())) {
                    for (const auto &neuron: layer) {
                        for (const auto &connection: make_iterator_range(
                                 connectionsTo(neuron))) {
                            f(connection);
                        }
                    }
                }
            }


            /*!
             * \brief Adds a layer to the neural network.
             *
             * The layer is appended to the end of the network and becomes
             * the new output layer. If it is the first layer, it also becomes
             * the input layer. If you do not want to follow this
             * logic, you can always explicitly set the input and
             * output layer.
             *
             * This method does not connect the bias neuron.
             *
             * \sa #inputLayer()
             * \sa #outputLayer()
             * \sa #biasNeuron()
             */
            NeuralNetwork &operator <<(Layer *layer);


            /*!
             * Returns the size of the neural network in terms of
             * the number of layers. Useful in combination with the
             * index operator.
             *
             * \return The number of layers this network contains
             *
             * \sa #operator[]
             */
            size_type size() const
            {
                return m_layers.size();
            }


            /*!
             * \brief Returns the layer at the designated index.
             *
             * Requires index < NeuralNetwork#size()
             *
             * \return The Layer at the given index; the
             *  result of using this method with index >= NeuralNetwork#size()
             *  is undefined.
             */
            Layer *layerAt(const size_type &index) const;


            /*!
             * \brief Returns the layer at the designated index.
             *
             * Requires index < NeuralNetwork#size()
             *
             * \return The Layer at the given index; the
             *  result of using this method with index >= NeuralNetwork#size()
             *  is undefined.
             */
            const Layer &operator [](const size_type &index) const;


            /*!
             * \brief Returns a reference to the Layer at the given index
             *
             * \param[in] index The position of the layer in the FIFO-sorted
             *  Layer list
             *
             * \return A reference to the layer at the given position. The
             *  result of using this method with index >= NeuralNetwork#size()
             *  is undefined.
             */
            Layer &operator [](const size_type &index);


            /*!
             * \brief Returns the input layer
             *
             * \return The input layer
             */
            Layer &inputLayer();


            /*!
             * \brief Returns the output layer
             *
             * \return The output layer
             */
            Layer &outputLayer();


            /*!
             * \brief Configures this neural network based on a pattern.
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
             * \brief Calculates the transition of values from one layer
             *  to another.
             *
             * Given the output of one layer, the `from` layer, this method
             * calculates the input to the destination, the `to` layer, as
             * it results from the connections of the neurons originating in
             * `from` leading to the neurons in `to`.
             *
             * This method *does not* take the BIAS neuron into account.
             * The bias neuron is considered only in ::calculateLayer().
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
            Vector calculateLayerTransition(
                    const Layer &from,
                    const Layer &to,
                    const Vector &input);


            /*!
             * \brief Activates a whole layer of neurons with the given
             *  input, taking the bias neuron into account.
             *
             * \param[in] layer The layer to activate
             *
             * \param[in] input The input to the layer's neurons
             *
             * \return The result of the activation of each neuron in the
             *  layer
             */
            Vector calculateLayer(Layer &layer, Vector const& input);


            /*!
             * \brief Calculates a complete pass of the neural network.
             *
             * \input[in] input The input to the neural network,
             *  which is a vector that maps 1:1 an input value to an
             *  input neuron. If the size of the vector does not match
             *  the number of input neurons, an exception is thrown.
             */
            Vector calculate(const Vector &input);


            //! Clears the neural network completely
            void clear() override;


            /*!
             * \brief Initializes (deserializes) the ANN from its JSON
             *  representation
             *
             * \param[in] json The ANNs JSON representation
             */
            void fromJSON(QJsonDocument const& json) override;


            /*!
             * \brief Serializes the ANN to JSON
             *
             * \return The JSON representation of the ANN
             */
            QJsonDocument toJSON() const override;


            //! Checks for equality of two ANNs.
            bool operator ==(const NeuralNetwork& other) const;


            //! Checks for inequality of two ANNs.
            bool operator !=(const NeuralNetwork& other) const;


        protected:


            //! Internal logger
            log4cxx::LoggerPtr logger;


        private:


            //! \brief The bias neuron
            std::unique_ptr<Neuron> m_biasNeuron;


            //! All Layers contained in this Neural Network
            boost::ptr_vector<Layer> m_layers;


            /*!
             * \brief A hash that indexes all connection originating
             *  from a certain neuron.
             */
            NeuronConnectionsMap m_connectionSources;


            /*!
             * \brief A hash that indexes all connections that lead to a
             *  certain neuron.
             */
            NeuronConnectionsMap m_connectionDestinations;


            /*!
             * \brief The pattern that is used to construct the network
             *  and calculate a run through it.
             *
             * Patterns do two things: First, they define the layout
             * of a network, and second, they define how a network
             * works when values are calculated with it.
             */
            std::unique_ptr<NeuralNetworkPattern> m_pattern;
        };
    } /* namespace ANN */
} /* namespace Winzent */


//! Appends the ANN's JSON representation to the text stream
QTextStream& operator <<(
        QTextStream &out,
        const Winzent::ANN::NeuralNetwork &network);


#endif /* NEURALNETWORK_H_ */
