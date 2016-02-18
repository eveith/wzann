/*!
 * \file	BasicException.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef BASICEXCEPTION_H_
#define BASICEXCEPTION_H_

#include <string>
#include <QException>


namespace Winzent {
    namespace ANN {


        class Neuron;


        /*!
         * Base class for all exceptions that happen within the
         * <code>ANN</code> namespace.
         */
        class BasicException: public QException
        {

        public:


            void raise() const {
                throw *this;
            }


            BasicException *clone() const {
                return new BasicException(*this);
            }
        };


        /*!
         * Represents exceptions which are thrown in cases where
         * a network operates on a layer of one certain size, while
         * the user requests a layer of another size.
         *
         * For example,
         * consider the NeuralNetwork#calculate method: Here, a
         * vector is supplied whose size must match the size of the
         * layer (i.e., one value of the vector per one neuron of the
         * input layer). If these do not match, an instance of this
         * exception is thrown.
         */
        class LayerSizeMismatchException: public BasicException
        {

        public:


            /*!
             * The actual size of the layer
             */
            int actualSize;


            /*!
             * The layer size which was assumed by the caller
             */
            int expectedSize;


            LayerSizeMismatchException(int actualSize, int expectedSize);


            virtual const char *what() const noexcept override;


        private:


            //! \brief Explanatory string
            std::string m_what;
        };


        /*!
         * Thrown when one wants to access a connection between two
         * neurons when actually no connection exists.
         */
        class NoConnectionException: public BasicException
        {

        };


        /*!
         * Thrown whenever the caller wants to modify a weight value
         * and the weight is set to fixed.
         */
        class WeightFixedException: public BasicException
        {
        };


        /*!
         * Thrown whenever a neuron is unknown in the current context. For
         * example, if a neuron should be connected to another one, but the
         * neuron is not part of the particular neural network.
         */
        class UnknownNeuronException: public BasicException
        {
            public:


            /*!
             * The neuron that was not known in the current context.
             */
            const Neuron* neuron;


            UnknownNeuronException(const Neuron *unknownNeuron):
                    neuron(unknownNeuron) {}
        };

    } /* namespace ANN */
} /* namespace Winzent */


#endif /* BASICEXCEPTION_H_ */
