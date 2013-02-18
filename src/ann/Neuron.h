/*!
 * \file	Neuron.h
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */

#ifndef NEURON_H_
#define NEURON_H_


#include <QObject>


class QTextStream;


namespace Winzent
{
    namespace ANN
    {
        class ActivationFunction;
        class NeuralNetwork;


        class Neuron: public QObject
        {
            Q_OBJECT

            friend QTextStream& operator<<(QTextStream&, const NeuralNetwork&);

        private:


            /*!
             * The activation function used for neuron activation
             *
             * \sa #activate
             */
            ActivationFunction *m_activationFunction;


            /*!
             * Caches that last input that was presented to this neuron.
             */
            double m_lastInput;


            /*!
             * Caches the result of the last activation
             */
            double m_lastResult;


        public:


            /*!
             * Creates a new neuron with a specific activation
             * function
             *
             * \param activationFunction The activation function to
             *  use
             * \param parent The parent object
             *
             * \sa #activate
             * \sa QObject#setParent
             */
            Neuron(ActivationFunction *activationFunction, QObject *parent = 0);


            /*!
             * Copy constructor
             */
            Neuron(const Neuron &rhs);


            /*!
             * The constructor does not call <code>delete</code>
             * on the supplied activation function since it might be
             * shared with other neurons.
             */
            virtual ~Neuron();


            /*!
             * Clones this neuron by creating a new one with the
             * same activation function and the same last result.
             *
             * \return A new neuron with the same state as this one
             *
             * \sa ActivationFunction#clone
             * \sa #lastResult
             */
            Neuron* clone() const;


            /*!
             * Returns the last network input for this neuron
             */
            double lastInput() const;


            /*!
             * Returns the result of the last activation
             */
            double lastResult() const;


            /*!
             * Returns the activation function this neuron instance uses.
             */
            ActivationFunction* activationFunction() const;


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
