#ifndef WINZENT_ANN_CONNECTION_H
#define WINZENT_ANN_CONNECTION_H

#include <QObject>

#include "Exception.h"


namespace Winzent {
    namespace ANN {

        class Neuron;
        

        /*!
         * Instances of this class represent a connection between two neurons.
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
        class Connection: public QObject
        {
            Q_OBJECT

        private:


            /*!
             * The weight that is attached to this connection.
             */
            double m_weight;


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
             * Creates a new connection
             */
            explicit Connection(
                    Neuron *source,
                    Neuron *destination,
                    double weight,
                    QObject *parent = 0);


            /*!
             * Copy constructor
             */
            Connection(const Connection &rhs);


            /*!
             * Creates a clone of this connection.
             */
            Connection *clone() const;


            /*!
             * Returns the current weight attached to this connection.
             */
            double weight() const;


            /*!
             * Sets a new weight value.
             */
            void weight(double weight) throw(WeightFixedException);


            /*!
             * Sets the weight to a random value between
             * <code>min</code> and <code>max</code>.
             *
             * \input[in] min The minimum inclusive value
             *
             * \input[in] max The maximum inclusive value
             *
             * \return The random value the weight was set to.
             *
             * \throws WeightFixedException if the weight is fixed.
             */
            double setRandomWeight(const double &min, const double &max)
                    throw(WeightFixedException);


            /*!
             * Returns whether the weight associated with this connection is
             * fixed or not.
             */
            bool fixedWeight() const;


            /*!
             * Sets the connection weight fixed (i.e., untrainable) or variable.
             */
            void fixedWeight(bool fixed);


            /*!
             * Returns the current source neuron.
             */
            const Neuron *source() const;


            /*!
             * Sets a new source neuron.
             */
            void source(Neuron *source);


            /*!
             * Returns the current destination neuron.
             */
            Neuron *destination() const;


            /*!
             * Sets a new destination neuron.
             */
            void destination(Neuron *destination);


            /*!
             * Multiplies a value with the weight of this connection and returns
             * the result.
             */
            double operator *(const double &rhs) const;

        signals:
            
        public slots:
            
        };
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_CONNECTION_H
