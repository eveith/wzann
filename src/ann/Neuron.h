/*!
 * \file	Neuron.h
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */

#ifndef NEURON_H_
#define NEURON_H_


#include <QVector>

#include <memory>


class QTextStream;


using std::shared_ptr;


namespace Winzent {
    namespace ANN {


        class Layer;
        class NeuralNetwork;
        class ActivationFunction;


        class Neuron
        {
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
            std::shared_ptr<ActivationFunction> m_activationFunction;


            /*!
             * \brief Caches the input that was presented to #activate().
             */
            double m_lastInput;


            /*!
             * \brief Caches the result of the last activation
             */
            double m_lastResult;


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
            Neuron(ActivationFunction *activationFunction);


            /*!
             * \brief Constructs a new Neuron with a shared activation
             *  function
             *
             * \param[in] activationFunction The activation function for this
             *  neuron that is shared with other neurons
             */
            Neuron(shared_ptr<ActivationFunction> &activationFunction);


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
            double lastInput() const;


            /*!
             * \return All cached inputs
             */
            const QVector<double> lastInputs() const;


            /*!
             * Returns the result of the last activation
             */
            double lastResult() const;


            /*!
             * \return All cached results
             */
            const QVector<double> lastResults() const;


            /*!
             * \return The current size of the last input/last result caches.
             *
             * \deprecated
             */
            int cacheSize() const;


            /*!
             * \brief Sets the new input/result cache size.
             *
             * \return `this`
             *
             * \deprecated
             */
            Neuron &cacheSize(const int &cacheSize);


            /*!
             * \brief Returns the activation function
             *  this neuron instance uses.
             */
            ActivationFunction *activationFunction() const;


            /*!
             * \brief Sets a new activation function
             *
             * \param[in] activationFunction The activation function
             *
             * \return `*this`
             */
            Neuron &activationFunction(
                    ActivationFunction *const &activationFunction);

            /*!
             * \brief Sets a new activation function
             *
             * Invoking this method explicitly allows to share an
             * ActivationFunction object with other neurons.
             *
             * \param[in] activationFunction The activation function
             *
             * \return `*this`
             */
            Neuron &activationFunction(
                    shared_ptr<ActivationFunction> &activationFunction);


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
            double activate(const double &sum);
        };
    } /* namespace ANN */
} /* namespace Winzent */

#endif /* NEURON_H_ */
