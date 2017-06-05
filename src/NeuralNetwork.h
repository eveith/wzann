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

#include <Variant/Variant.h>
#include <Variant/VariantDefines.h>

#include "Layer.h"
#include "Neuron.h"
#include "Vector.h"
#include "Connection.h"
#include "JsonSerializable.h"
#include "LibVariantSupport.h"
#include "NeuralNetworkPattern.h"


using boost::ptr_vector;


namespace Winzent {
    namespace ANN {
        class TrainingSet;
        class TrainingAlgorithm;
        class NeuralNetworkPattern;


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
        class NeuralNetwork
        {
            friend class NeuralNetworkPattern;
            friend class AbstractTrainingStrategy;

            friend libvariant::Variant to_variant<>(NeuralNetwork const&);
            friend NeuralNetwork from_variant<>(libvariant::Variant const&);


        public:


            typedef std::size_t size_type;

            typedef ptr_vector<Layer>::iterator LayerIterator;
            typedef ptr_vector<Layer>::const_iterator LayerConstIterator;
            typedef std::pair<
                    LayerIterator,
                    LayerIterator> LayerRange;
            typedef std::pair<
                    LayerConstIterator,
                    LayerConstIterator> LayerConstRange;

            typedef std::vector<Connection*> ConnectionsPtrVector;
            typedef ConnectionsPtrVector::iterator
                    ConnectionPtrIterator;
            typedef ConnectionsPtrVector::const_iterator
                    ConnectionPtrConstIterator;
            typedef std::pair<
                    ConnectionPtrIterator,
                    ConnectionPtrIterator> ConnectionPtrRange;
            typedef std::pair<
                    ConnectionPtrConstIterator,
                    ConnectionPtrConstIterator> ConnectionPtrConstRange;

            typedef std::unordered_map<
                    Neuron*,
                    ConnectionsPtrVector> NeuronConnectionsMap;

            typedef boost::ptr_vector<Connection>::iterator
                    ConnectionIterator;
            typedef boost::ptr_vector<Connection>::const_iterator
                    ConnectionConstIterator;
            typedef std::pair<
                    ConnectionIterator,
                    ConnectionIterator> ConnectionRange;
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
            NeuralNetwork* clone() const;


            /*!
             * \brief Returns this network's bias neuron
             *
             * \return The bias neuron of this Neural Network
             */
            Neuron const& biasNeuron() const;


            /*!
             * \brief Returns a modifiable version of the network's bias neuron
             *
             * \return The bias neuron, modifiable
             */
            Neuron& biasNeuron();


            /*!
             * Checks whether a certain neuron is part of this
             * network or not.
             *
             * \param neuron The neuron to look for
             *
             * \return <code>true</code> if it lives in this network,
             *  <code>false</code> otherwise.
             */
            bool contains(Neuron const& neuron) const;


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
            bool connectionExists(Neuron const& from, Neuron const& to) const;


            /*!
             * Connects two neurons. Does not set a weight.
             *
             * \param from The neuron the connection originates
             *
             * \param to The destination neuron
             *
             * \return The new connection
             */
            Connection& connectNeurons(Neuron const& from, Neuron const& to);


            /*!
             * \brief Removes the connection between two neurons
             *
             * \param[in] from The source neuron
             *
             * \param[in] to The destination neuron
             */
            void disconnectNeurons(Neuron const& from, Neuron const& to);


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
            Connection* connection(Neuron const& from, Neuron const& to)
                    const;


            /*!
             * \brief Provides an const interator range over all connections
             *  in this neural network
             *
             * The connections appear every time in the same order.
             *
             * \return A range of const iterators over all connections.
             */
            ConnectionPtrRange connections();


            /*!
             * \brief Provides an const interator range over all connections
             *  in this neural network
             *
             * The connections appear every time in the same order.
             *
             * \return A range of const iterators over all connections.
             */
            ConnectionPtrConstRange connections() const;


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
            ConnectionPtrRange connectionsFrom(Neuron const& neuron);


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
            ConnectionPtrConstRange connectionsFrom(Neuron const& neuron)
                    const;


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
            ConnectionPtrRange connectionsTo(const Neuron &neuron);


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
            ConnectionPtrConstRange connectionsTo(const Neuron &neuron) const;


            /*!
             * \brief Allows to iterate over all layers in the ANN
             *
             * \return A pair of layer iterators: `[begin, end)`
             */
            LayerRange layers();


            /*!
             * \brief Allows to iterate over read-only-accessible layers
             *
             * \return A pair of const iterators over all layers
             */
            LayerConstRange layers() const;


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
            NeuralNetwork& operator <<(Layer* layer);


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
            Layer* layerAt(size_type index) const;


            /*!
             * \brief Returns the layer at the designated index.
             *
             * Requires index < NeuralNetwork#size()
             *
             * \return The Layer at the given index; the
             *  result of using this method with index >= NeuralNetwork#size()
             *  is undefined.
             */
            Layer const& operator [](size_type index) const;


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
            Layer& operator [](size_type index);


            /*!
             * \brief Returns the input layer
             *
             * \return The input layer
             */
            Layer& inputLayer();


            /*!
             * \brief Returns the output layer
             *
             * \return The output layer
             */
            Layer& outputLayer();


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
            NeuralNetwork& configure(NeuralNetworkPattern const& pattern);


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
                    Layer const& from,
                    Layer const& to,
                    Vector const& input);


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
            Vector calculate(Vector const& input);


            //! Checks for equality of two ANNs.
            bool operator ==(const NeuralNetwork& other) const;


            //! Checks for inequality of two ANNs.
            bool operator !=(const NeuralNetwork& other) const;


        private:


            //! \brief The bias neuron
            std::unique_ptr<Neuron> m_biasNeuron;


            //! \brief All Layers contained in this NeuralNetwork
            boost::ptr_vector<Layer> m_layers;


            //! \brief All connections between neurons in this NeuralNetwork
            ConnectionsPtrVector m_connections;


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


        template <>
        inline libvariant::Variant to_variant(NeuralNetwork const& network)
        {
            libvariant::Variant o;

            o["version"] = NeuralNetwork::VERSION;
            o["biasNeuron"] = to_variant(*(network.m_biasNeuron));

            libvariant::Variant::List layers;
            for (auto const& layer: network.m_layers) {
                layers.push_back(to_variant(layer));
            }
            o["layers"] = layers;

            libvariant::Variant::List connections;
            for (auto const& connectionSources: network.m_connectionSources) {
                for (auto const& c: connectionSources.second) {
                    int srcLayer = -1,
                            dstLayer = -1,
                            srcNeuron = -1,
                            dstNeuron = -1;
                    libvariant::Variant connection;

                    for (NeuralNetwork::size_type i = 0;
                            i != network.m_layers.size(); ++i) {
                        for (NeuralNetwork::size_type j = 0;
                                j != network.m_layers.at(i).size(); ++j) {
                            auto const& n = network.layerAt(i)->neuronAt(j);

                            if (n == &(c->source())) {
                                srcLayer = i;
                                srcNeuron = j;
                            } else if (n == &(c->destination())) {
                                dstLayer = i;
                                dstNeuron = j;
                            }
                        }
                    }

                    connection["srcLayer"] = srcLayer;
                    connection["srcNeuron"] = srcNeuron;
                    connection["dstLayer"] = dstLayer;
                    connection["dstNeuron"] = dstNeuron;
                    connection["weight"] = c->weight();
                    connection["fixedWeight"] = c->fixedWeight();

                    if (-1 == srcNeuron) {
                        connection["srcNeuron"] = "BIAS";
                    }

                    connections.push_back(connection);
                }
            }
            o["connections"] = connections;

            o["pattern"] = libvariant::Variant(
                    libvariant::VariantDefines::NullType);
            if (network.m_pattern != nullptr) {
                o["pattern"] = to_variant(*(network.m_pattern));
            }

            return o;
        }


        template <>
        inline NeuralNetwork from_variant(libvariant::Variant const& variant)
        {
            NeuralNetwork ann;

            ann.m_biasNeuron.reset(new_from_variant<Neuron>(
                    variant["biasNeuron"]));

            auto const& layers = variant["layers"].AsList();
            for (auto const& i: layers) {
                ann << new_from_variant<Layer>(i);
            }

            auto const& connections = variant["connections"].AsList();
            for (auto const& c: connections) {
                auto &src = (c["srcNeuron"] == "BIAS"
                        ? ann.biasNeuron()
                        : ann[c["srcLayer"].AsUnsigned()][
                            c["srcNeuron"].AsUnsigned()]);
                auto &dst = ann[c["dstLayer"].AsUnsigned()][
                        c["dstNeuron"].AsUnsigned()];

                ann.connectNeurons(src, dst)
                    .weight(c["weight"].AsDouble())
                    .fixedWeight(c["fixedWeight"].AsBool());
            }

            if (variant.Contains("pattern")
                    && variant["pattern"].GetType()
                        != libvariant::VariantDefines::NullType) {
                ann.m_pattern.reset(new_from_variant<NeuralNetworkPattern>(
                        variant["pattern"]));
                assert(ann.m_pattern != nullptr);
            }

            return ann;
        }


        template <> inline NeuralNetwork*
        new_from_variant(libvariant::Variant const& variant)
        {
            return new NeuralNetwork(from_variant<NeuralNetwork>(variant));
        }
    } /* namespace ANN */
} /* namespace Winzent */


#endif /* NEURALNETWORK_H_ */
