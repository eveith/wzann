/*!
 * \file	NeuralNetworkPattern.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef NEURALNETWORKPATTERN_H_
#define NEURALNETWORKPATTERN_H_


#include <initializer_list>

#include <QPair>
#include <QObject>
#include <QJsonDocument>

#include <JsonSerializable.h>

#include "NeuralNetwork.h"
#include "Winzent-ANN_global.h"


class QString;


using std::initializer_list;


namespace Winzent {
    namespace ANN {


        class ActivationFunction;


        class WINZENTANNSHARED_EXPORT NeuralNetworkPattern:
                public QObject,
                public JsonSerializable
        {
            Q_OBJECT


            friend Vector NeuralNetwork::calculate(const Vector &);
            friend NeuralNetwork &NeuralNetwork::configure(
                    const NeuralNetworkPattern &);


        public:


            typedef QList<int> LayerSizes;
            typedef QList<ActivationFunction*> ActivationFunctions;


            /*!
             * \brief Defines a Layer in terms of number of neurons and
             *  ActivationFunction for each Layer.
             */
            typedef QPair<int, ActivationFunction const&> LayerDefinition;


            //! \brief Creates a new, empty pattern.
            NeuralNetworkPattern();


            /*!
             * Constructs a new pattern and supplies the sizes of the
             * layers.
             *
             * \param layerSizes The size of each layer; which
             *  semantics apply depends on the actual, derived
             *  pattern class which is used.
             *
             * \param activationFunctions The activation functions
             *  that apply to each layer.
             */
            NeuralNetworkPattern(
                    LayerSizes const& layerSizes,
                    ActivationFunctions const& activationFunctions);


            /*!
             * Constructs a new pattern, but uses C++11
             * <code>initializer_list</code>s.
             *
             * Example:
             *
             *      NeuralNetworkPattern({
             *              2,
             *              6,
             *              2
             *          }, {
             *              new SigmoidActivationFunction(),
             *              new SigmoidActivationFunction(),
             *              new SigmoidActivationFunction()
             *          });
             *
             * \param[in] layerSizes The size of each layer; which
             *  semantics apply depends on the actual, derived
             *  pattern class which is used.
             *
             * \param[in] activationFunctions The activation functions
             *  that apply to each layer.
             */
            NeuralNetworkPattern(
                    initializer_list<int> layerSizes,
                    initializer_list<ActivationFunction *>
                        activationFunctions);



            /*!
             * Clones the instance of the derived pattern.
             *
             * \return A complete clone of the instance
             */
            virtual NeuralNetworkPattern *clone() const = 0;


            /*!
             * \brief The destructor will also delete all ActivationFunction
             *  objects supplied for configuring the network.
             */
            virtual ~NeuralNetworkPattern();


            /*!
             * \brief Adds the definition of a layer to the Pattern
             *
             * \param[in] layerDefinition Definition of the layer to add:
             *  size and ActivationFunction
             *
             * \return `*this`
             */
            NeuralNetworkPattern& add(LayerDefinition const& layerDefinition);


            //! Clears the pattern completely.
            void clear() override;


            /*!
             * \brief Serializes the pattern to JSON
             *
             * \return The pattern's JSON representation
             */
            QJsonDocument toJSON() const override;


            /*!
             * \brief Deserializes the pattern from JSON
             *
             * \param[in] json The pattern's JSON representation
             */
            void fromJSON(const QJsonDocument &json) override;


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
            virtual bool equals(const NeuralNetworkPattern* const& other)
                    const;


        protected:


            /*!
             * A handy way to store the size of each layer. What index
             * corresponds to which layer and what semantic is
             * attached to each layer is not defined here, but depends
             * on the concrete pattern.
             */
            LayerSizes m_layerSizes;


            /*!
             * Lists the activation function per layer. The index
             * corresponds to the index of the #m_layerSizes member.
             */
            ActivationFunctions m_activationFunctions;


            /*!
             * Shortcut method that fully connects two layers of an
             * neural network.
             *
             * \param network The network that contains the layers
             *
             * \param fromLayer The originating layer
             *
             * \param toLayer The layer that contains the target
             *  neurons
             */
            void fullyConnectNetworkLayers(
                    NeuralNetwork *network,
                    const int &fromLayer,
                    const int &toLayer);


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
                    const Vector &input) = 0;


            /*!
             * \brief Configures any network to the layout the pattern
             *  represents. The layer sizes are given in the
             *  constructor; the rest of the layout is created by
             *  this method.
             *
             * \param[inout] network The neural network to configure
             */
            virtual void configureNetwork(NeuralNetwork *const &network) = 0;
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURALNETWORKPATTERN_H_ */
