/*!
 * \file	NeuralNetworkPattern.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef NEURALNETWORKPATTERN_H_
#define NEURALNETWORKPATTERN_H_


#include <list>
#include <typeinfo>

#include <boost/core/demangle.hpp>

#include "ClassRegistry.h"
#include "NeuralNetwork.h"
#include "LibVariantSupport.h"
#include "ActivationFunction.h"


using std::pair;


namespace Winzent {
    namespace ANN {


        class NeuralNetworkPattern
        {
            friend Vector NeuralNetwork::calculate(const Vector &);
            friend NeuralNetwork &NeuralNetwork::configure(
                    const NeuralNetworkPattern &);
            friend libvariant::Variant to_variant<>(
                    NeuralNetworkPattern const&);

        public:


            /*!
             * \brief A simple layer definition that associates an activation
             *  function to a number of neurons uniformly, i.e., each
             *  of the n neurons uses the same type of activation function.
             *
             * \sa ActivationFunction
             */
            typedef pair<unsigned, ActivationFunction> SimpleLayerDefinition;


            typedef std::list<SimpleLayerDefinition> LayerDefinitionList;


            //! \brief Creates a new, empty pattern.
            NeuralNetworkPattern();


            virtual ~NeuralNetworkPattern();


            /*!
             * \brief Clones the (derived) NeuralNetworkPattern object
             *
             * \return A complete clone of the instance
             */
            virtual NeuralNetworkPattern* clone() const = 0;


            /*!
             * \brief Adds the definition of a layer to the Pattern
             *
             * \param[in] layerDefinition Definition of the layer to add.
             *
             * \return `*this`, for chaining.
             *
             * \sa SimpleLayerDefinition
             */
            NeuralNetworkPattern& addLayer(
                    SimpleLayerDefinition layerDefinition);


            /*!
             * \brief Checks for equality between two patterns.
             *
             * Two patterns are equal if their number of layers and types of
             * activation functions match. However, in the base class, there
             * is no check whether the type of the pattern matches. This is
             * why derived classes must overload this method and add the
             * corresponding check, e.g., by trying to cast.
             *
             * \param[in] other The other pattern
             *
             * \return True if the two patterns are equal
             */
            virtual bool operator ==(NeuralNetworkPattern const& other) const;


            //! \brief Equivalent to `! (*this == other)`
            bool operator !=(NeuralNetworkPattern const& other) const;


        protected:


            LayerDefinitionList m_layerDefinitions;


            /*!
             * \brief Shortcut method that fully connects two layers of an
             *  neural network.
             *
             * \param from The originating layer
             *
             * \param to The layer that contains the target neurons
             */
            void fullyConnectNetworkLayers(Layer& from, Layer& to);


            /*!
             * Runs a vector of values through the neural network and
             * returns its result. The input vector size must match
             * the neural network's input layer size.
             *
             * \param network The network that is used for
             *  the calculation
             *
             * \param input The input values
             *
             * \return The result of the calculation
             */
            virtual Vector calculate(
                    NeuralNetwork &network,
                    Vector const& input) = 0;


            /*!
             * \brief Configures any network to the layout the pattern
             *  represents. The layer sizes are given in the
             *  constructor; the rest of the layout is created by
             *  this method.
             *
             * \param[inout] network The neural network to configure
             */
            virtual void configureNetwork(NeuralNetwork& network) = 0;
        };


        template <>
        inline libvariant::Variant to_variant(
                NeuralNetworkPattern::SimpleLayerDefinition const& d)
        {
            return libvariant::Variant({ d.frist, d.second });
        }


        template <>
        inline libvariant::Variant to_variant(
                NeuralNetworkPattern const& neuralNetworkPattern)
        {
            libvariant::Variant variant;

            variant["type"] = boost::core::demangle(
                    typeinfo(neuralNetworkPattern).name());
            variant["layerDefinitions"] = to_variant(
                    neuralNetworkPattern.m_layerDefinitions);

            return variant;
        }


        template <>
        inline NeuralNetworkPattern* new_from_variant(
                libvariant::Variant const& variant)
        {
            auto* pattern = ClassRegistry<NeuralNetworkPattern>::instance()
                    ->create(variant["type"].AsString());
            pattern->m_layerDefinitions = variant["layerDefinitions"]
                    .AsList();
            return pattern;
        }
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURALNETWORKPATTERN_H_ */
