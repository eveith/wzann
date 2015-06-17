#ifndef WINZENT_ANN_CONNECTION_H
#define WINZENT_ANN_CONNECTION_H

#include <QtGlobal>

#include "Exception.h"
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
         */
        class WINZENTANNSHARED_EXPORT Connection
        {
        private:


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


        public:


            /*!
             * \brief Creates a new connection
             */
            explicit Connection(
                    Neuron *const &source = nullptr,
                    Neuron *const &destination = nullptr,
                    const double &weight = 0.0);


            Connection(const Connection &) = delete;
            Connection(Connection &&) = delete;


            /*!
             * \brief Creates a clone of this connection.
             */
            Connection *clone() const;


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
            Connection &weight(const double &weight);


            /*!
             * Returns whether the weight associated with this connection is
             * fixed or not.
             */
            bool fixedWeight() const;


            /*!
             * Sets the connection weight fixed (i.e., untrainable) or variable.
             */
            void fixedWeight(const bool &fixed);


            /*!
             * Returns the current source neuron.
             */
            Neuron *source() const;


            /*!
             * \brief Sets a new source neuron.
             *
             * \return `*this`
             */
            Connection &source(Neuron *const &source);


            /*!
             * \brief Returns the current destination neuron.
             *
             * \return The connection's destination
             */
            Neuron *destination() const;


            /*!
             * \brief Sets a new destination neuron.
             *
             * \return `*this`
             */
            Connection &destination(Neuron *const &destination);


            /*!
             * \brief Multiplies a value with the weight of this connection
             *
             * \param[in] rhs The right-hand side operand
             *
             * \return this->weight() * rhs
             */
            double operator *(const double &rhs) const;
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONNECTION_H
