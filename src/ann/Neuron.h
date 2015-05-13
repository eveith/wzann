/*!
 * \file	Neuron.h
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */

#ifndef NEURON_H_
#define NEURON_H_


#include <QObject>
#include <QVector>


class QTextStream;


namespace Winzent
{
    namespace ANN
    {
        class Layer;
        class NeuralNetwork;
        class ActivationFunction;


        class Neuron: public QObject
        {
            Q_OBJECT

            friend class Layer;
            friend QTextStream& operator<<(QTextStream&, const NeuralNetwork&);

        private:


            /*!
             * \brief Our parent layer
             */
            Layer *m_parent;


            /*!
             * \brief The activation function used for neuron activation
             *
             * \sa #activate()
             */
            ActivationFunction *m_activationFunction;


            /*!
             * \brief Caches the input that was presented to #activate().
             */
            qreal m_lastInput;


            /*!
             * \brief Caches the result of the last activation
             */
            qreal m_lastResult;


        public:


            /*!
             * \brief Creates a new neuron with a specific activation
                 * function
             *
             * \param activationFunction The activation function that is used to
             *  calculation the neuron's activation. The Neuron object takes
             *  ownership of the activation function object.
             *
             * \param parent The parent object
             *
             * \sa #activate
             *
             * \sa QObject#setParent
             */
            Neuron(ActivationFunction *activationFunction, QObject *parent = 0);


            /*!
             * \brief Deleted copy constructor
             *
             * No copy constructor exists for the Neuron class. If you need
             * a deep copy of a neuron, call #clone() instead.
             *
             * \sa Neuron#clone()
             */
            Neuron(const Neuron &) = delete;

            Neuron(Neuron &&) = delete;


            /*!
             * Clones this neuron by creating a new one with the
             * same activation function and the same last result.
             *
             * \return A new neuron with the same state as this one
             *
             * \sa ActivationFunction#clone
             * \sa #lastResult
             */
            Neuron *clone() const;


            /*!
             * \brief Returns the parent Layer
             *
             * \return The parent layer, or `nullptr` if the Neuron does not
             *  belong to any Layer
             *
             * \sa Layer#addNeuron()
             */
            Layer *parent() const;


            /*!
             * Returns the last network input for this neuron
             */
            qreal lastInput() const;


            /*!
             * \return All cached inputs
             */
            const QVector<qreal> lastInputs() const;


            /*!
             * Returns the result of the last activation
             */
            qreal lastResult() const;


            /*!
             * \return All cached results
             */
            const QVector<qreal> lastResults() const;


            /*!
             * \return The current size of the last input/last result caches.
             */
            int cacheSize() const;


            /*!
             * Sets the new input/result cache size.
             *
             * \return `this`
             */
            Neuron &cacheSize(const int &cacheSize);


            /*!
             * Returns the activation function this neuron instance uses.
             */
            ActivationFunction *activationFunction() const;


            /*!
             * Activates the neuron given the input sum of all
             * weights that lead to this neuron. Also stores the
             * result, which can be re-retrievend using #lastResult.
             *
             * \param[in] sum   The input, i.e. the sum of all
             *  weighted inputs of all connections leading to this
             *  neuron.
             *
             * \return The result of the activation
             *
             * \sa #lastResult
             * \sa #m_activationFunction
             */
            qreal activate(const qreal &sum);
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURON_H_ */
