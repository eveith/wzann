/*!
 * \file	BasicException.h
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#ifndef BASICEXCEPTION_H_
#define BASICEXCEPTION_H_

#include <qtconcurrentexception.h>

namespace Winzent
{
    namespace ANN
    {

        /*!
         * Base class for all exceptions that happen within the
         * <code>ANN</code> namespace.
         */
        class BasicException: public QtConcurrent::Exception
        {

        public:


            void raise() const {
                throw *this;
            }


            Exception *clone() const {
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


            LayerSizeMismatchException(int actualSize, int expectedSize):
                    BasicException(),
                    actualSize(actualSize),
                    expectedSize(expectedSize) {}

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

    } /* namespace ANN */
} /* namespace Winzent */


#endif /* BASICEXCEPTION_H_ */
