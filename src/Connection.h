#ifndef WINZENT_ANN_CONNECTION_H
#define WINZENT_ANN_CONNECTION_H


#include "NeuralNetwork.h"


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
        class Connection
        {
        public:


            Connection() = delete;
            Connection(const Connection &) = delete;
            Connection(Connection &&) = delete;


            /*!
             * \brief Returns the current weight attached to this connection.
             *
             * \return The connection's weight
             */
            double weight() const;


            /*!
             * \brief Sets a new weight value.
             *
             * \throw WeightFixedException If the connection has a fixed
             *  weight
             *
             * \return `*this`
             */
            Connection& weight(double weight);


            /*!
             * \brief Indicates whether the weight associated
             *  with this connection is fixed or not.
             */
            bool fixedWeight() const;


            /*!
             * \brief Sets the connection weight fixed (i.e., untrainable)
             *  or variable.
             */
            Connection& fixedWeight(bool fixed);


            //! \brief The source neuron
            Neuron& source();


            //! \brief The source neuron
            const Neuron& source() const;


            //! \brief The destination neuron
            Neuron& destination();


            //! \brief The destination neuron
            const Neuron& destination() const;


            /*!
             * \brief Multiplies a value with the weight of this connection
             *
             * \param[in] rhs The right-hand side operand
             *
             * \return this->weight() * rhs
             */
            double operator *(double rhs) const;


            /*!
             * \brief Checks for connection identity
             *
             * Two connections are equal if they have the same weight, the
             * same "weight fixed" value, and their sources and destinations
             * are the same.
             *
             * \param[in] other The other connection
             *
             * \return True if the connections are equal as explained; false
             *  otherwise.
             */
            bool operator ==(Connection const& other) const;


            /*!
             * \brief Checks for connection inequality.
             *
             * \param[in] other The other connection object
             *
             * \return `! (*this == other)`
             *
             * \sa #operator==()
             */
            bool operator !=(Connection const& other) const;


        private:


            friend class NeuralNetwork;



            /*!
             * \brief Creates a new connection
             */
            Connection(
                    Neuron &source,
                    Neuron &destination,
                    double weight = 0.0);



            /*!
             * \brief Sets a new source neuron.
             *
             * \return `*this`
             */
            Connection& source(Neuron& source);



            /*!
             * \brief Sets a new destination neuron.
             *
             * \return `*this`
             */
            Connection& destination(Neuron& destination);


            /*!
             * \brief The weight that is attached to this connection
             */
            double m_weight;


            /*!
             * \brief Whether the weight value of this connection is fixed
             *  (i.e., cannot be trained) or may be changed.
             */
            bool m_fixed;


            /*!
             * \brief The neuron from which this connection originates
             */
            Neuron* m_sourceNeuron;


            /*!
             * \brief The destination neuron to which this connection leads
             */
            Neuron* m_destinationNeuron;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONNECTION_H
