/*!
 * \file	NeuralNetworkPattern.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef NEURALNETWORKPATTERN_H_
#define NEURALNETWORKPATTERN_H_


#include <initializer_list>

#include <qobject.h>
#include <QList>

#include "NeuralNetwork.h"


class QString;


using std::initializer_list;


namespace Winzent {
    namespace ANN {


        class ActivationFunction;


        class NeuralNetworkPattern
        {


            friend ValueVector NeuralNetwork::calculate(const ValueVector &);
            friend NeuralNetwork &NeuralNetwork::configure(
                    const NeuralNetworkPattern &);


        protected:


            /*!
             * A handy way to store the size of each layer. What index
             * corresponds to which layer and what semantic is
             * attached to each layer is not defined here, but depends
             * on the concrete pattern.
             */
            QList<int> m_layerSizes;


            /*!
             * Lists the activation function per layer. The index
             * corresponds to the index of the #m_layerSizes member.
             */
            QList<ActivationFunction *> m_activationFunctions;


            /*!
             * \sa ::weightRandomMin
             */
            qreal m_weightRandomMin;


            /*!
             * \sa ::weightRandomMax
             */
            qreal m_weightRandomMax;


            NeuralNetworkPattern():
                    m_weightRandomMin(weightRandomMin),
                    m_weightRandomMax(weightRandomMax)
            {
            }


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
            virtual ValueVector calculate(
                    NeuralNetwork *const &network,
                    const ValueVector &input) = 0;


            /*!
             * Configures any network to the layout the pattern
             * represents. The layer sizes are given in the
             * constructor; the rest of the layout is created by
             * this method.
             *
             * \param network The neural network to configure
             */
            virtual void configureNetwork(NeuralNetwork *network) = 0;


        public:


            /*!
             * Minimum value of random weights upon initialization.
             * This is the template value for all instances.
             *
             * \sa #m_weightRandomMin
             */
            static qreal weightRandomMin;


            /*!
             * Maximum value of random weights upon initialization.
             * This is the template value for all instances.
             *
             * \sa #m_weightRandomMax
             */
            static qreal weightRandomMax;


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
                    QList<int> layerSizes,
                    QList<ActivationFunction *> activationFunctions);


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
             * \param layerSizes The size of each layer; which
             *  semantics apply depends on the actual, derived
             *  pattern class which is used.
             *
             * \param activationFunctions The activation functions
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
             * Returns the string representation of this class. Needed
             * to dump neural networks to disk.
             */
            QString toString();
        };

    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURALNETWORKPATTERN_H_ */
