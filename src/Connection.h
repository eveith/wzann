#ifndef WINZENT_ANN_CONNECTION_H
#define WINZENT_ANN_CONNECTION_H


#include "NeuralNetwork.h"
#include "Winzent-ANN_global.h"


namespace Winzent {
    namespace ANN {

        class Neuron;


        /*!
         * \brief Instances of this class represent a connection
         *  between two neurons.
         *
         * A connection is always unidirectional, meaning it has a source and
         * a destination. If you need a two-ways connection, you'll need to
         * create two separate instances of this class.
         *
         * A connection has a weight attached to it which modifies any value
         * traveling via this particular connection. Normally, weights can be
         * trained, i.e. their value is modified. However, for certain ANN
         * layouts, fixed connections exist. As such, every connection object
         * has a boolean switch to make the weight fixed.
         *
         * Connection objects are created by ::NeuralNetwork objects through
         * their connect/disconnect methods; it is never created by an user
         * directly.
         *
         * \sa NeuralNetwork::connectNeurons()
         *
         * \sa NeuralNetwork::disconnectNeurons()
         */
        class WINZENTANNSHARED_EXPORT Connection
        {
        public:


            Connection() = delete;
            Connection(const Connection &) = delete;
            Connection(Connection &&) = delete;


            /*!
             * \brief Creates a clone of this connection.
             *
             * \return A clone of this connection
             */
            Connection *clone() const;


            /*!
             * \brief Returns the current weight attached to this connection.
             *
             * \return The connection's weight
             */
            qreal weight() const;


            /*!
             * \brief Sets a new weight value.
             *
             * \throw WeightFixedException If the connection has a fixed
             *  weight
             *
             * \return `*this`
             */
            Connection &weight(const qreal &weight);


            /*!
             * \brief Indicates whether the weight associated
             *  with this connection is fixed or not.
             */
            bool fixedWeight() const;


            /*!
             * \brief Sets the connection weight fixed (i.e., untrainable)
             *  or variable.
             */
            Connection &fixedWeight(const bool &fixed);


            //! \brief The source neuron
            Neuron &source();


            //! \brief The source neuron
            const Neuron &source() const;


            //! \brief The destination neuron
            Neuron &destination();


            //! \brief The destination neuron
            const Neuron &destination() const;


            /*!
             * \brief Multiplies a value with the weight of this connection
             *
             * \param[in] rhs The right-hand side operand
             *
             * \return this->weight() * rhs
             */
            qreal operator *(const qreal &rhs) const;


            /*!
             * \brief Checks whether two Connections are equivalent
             *
             * Two Connections are equivalent if their parameters
             * (weight, fixedWeight) are equals, and if their source and
             * destination neurons are also equivalent.
             *
             * This means that two connections can belong to different
             * networks, but fulfill the same purpose. If you want to find
             * out whether two connections really belong to the same network,
             * compare them using Connection::operator==().
             *
             * \param[in] other The other connection
             *
             * \return `true` if the two connections are equaivalent,
             *  `false` if not.
             *
             * \sa Connection::operator ==()
             *
             * \sa Neuron::equals()
             */
            bool equals(const Connection &other) const;


            /*!
             * \brief Checks for connection identity
             *
             * Two connections are equal if they have the same weight, the
             * same "weight fixed" value, and their sources and destinations
             * are the same. Here, source/destination equality is defined by
             * the Neuron::operator==() method.
             *
             * Thus, this method checks for Connection identity, because
             * the Neuron::operator ==() also checks for the neuron's
             * identity. To compare two objects parameter-wise, use
             * Connection::eqauls().
             *
             * \param[in] other The other connection
             *
             * \return True if the connections are equal as explained; false
             *  otherwise.
             */
            bool operator ==(const Connection& other) const;


            /*!
             * \brief Checks for connection inequality.
             *
             * \param[in] other The other connection object
             *
             * \return `! (*this == other)`
             *
             * \sa #operator==()
             */
            bool operator !=(const Connection& other) const;


        private:


            friend class NeuralNetwork;



            /*!
             * \brief Creates a new connection
             */
            Connection(
                    Neuron &source,
                    Neuron &destination,
                    const qreal &weight = 0.0);



            /*!
             * \brief Sets a new source neuron.
             *
             * \return `*this`
             */
            Connection &source(Neuron &source);



            /*!
             * \brief Sets a new destination neuron.
             *
             * \return `*this`
             */
            Connection &destination(Neuron &destination);


            /*!
             * The weight that is attached to this connection.
             */
            qreal m_weight;


            /*!
             * Whether the weight value of this connection is fixed
             * (i.e., cannot be trained) or may be changed.
             */
            bool m_fixed;


            /*!
             * The neuron from which this connection originates.
             */
            Neuron *m_sourceNeuron;


            /*!
             * The destination neuron to which this connection leads.
             */
            Neuron *m_destinationNeuron;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONNECTION_H
