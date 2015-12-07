/*!
 * \file
 */


#ifndef WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
#define WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H


#include <initializer_list>

#include <QObject>
#include <QList>

#include <ClassRegistry.h>

#include "NeuralNetwork.h"
#include "NeuralNetworkPattern.h"

#include "Winzent-ANN_global.h"


using std::initializer_list;


namespace Winzent {
    namespace ANN {

        class ActivationFunction;

        /*!
         * \brief Instances of this class represent the pattern to create a
         *  (potentially multi-layered), feed-forward perceptron without
         *  recurrent connections, shortcut connections or other specialities.
         */
        class WINZENTANNSHARED_EXPORT PerceptronNetworkPattern:
                public NeuralNetworkPattern
        {
            Q_OBJECT

            friend class Winzent::ClassRegistration<PerceptronNetworkPattern>;


        public:


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    QList<int> layerSizes,
                    QList<ActivationFunction *> activationFunctions);


            /*!
             * Creates a new perceptron pattern given the number of neuron
             * in the particular layers and their activation functions.
             *
             * \sa NeuralNetworkPattern#NeuralNetworkPattern
             */
            PerceptronNetworkPattern(
                    initializer_list<int> layerSizes,
                    initializer_list<ActivationFunction *>
                        activationFunctions);


            /*!
             * \brief Creates a clone of a pattern instance.
             *
             * \sa NeuralNetworkPattern#clone
             */
            virtual NeuralNetworkPattern *clone() const override;


            /*!
             * \brief Checks for equality of two PerceptronNetworkPatterns
             *
             * \param[in] other The other pattern
             *
             * \return True if the two are of the same class and have the
             *  same parameters
             */
            virtual bool equals(const NeuralNetworkPattern* const& other)
                    const override;


        protected:


            /*!
             * \brief Configures the supplied neural network
             *  to be an perceptron.
             */
            virtual void configureNetwork(NeuralNetwork *const &network)
                    override;


            /*!
             * \sa NeuralNetworkPattern#calculate
             */
            virtual Vector calculate(
                    NeuralNetwork &network,
                    const Vector &input)
                    override;


        private:


            PerceptronNetworkPattern();
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_PERCEPTRONNETWORKPATTERN_H
