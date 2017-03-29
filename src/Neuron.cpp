#include "Layer.h"
#include "ActivationFunction.h"

#include "Neuron.h"


using std::shared_ptr;


namespace Winzent {
    namespace ANN {
        Neuron::Neuron():
                m_parent(nullptr),
                m_activationFunction(ActivationFunction::Null)
        {
        }


        Neuron::~Neuron()
        {
        }


        Neuron* Neuron::clone() const
        {
            Neuron *n = new Neuron();

            n->m_parent = m_parent;
            n->m_lastInput = m_lastInput;
            n->m_lastResult = m_lastResult;
            n->m_activationFunction = m_activationFunction;

            return n;
        }


        Layer* Neuron::parent() const
        {
            return m_parent;
        }


        double Neuron::lastResult() const
        {
            return m_lastResult;
        }


        double Neuron::lastInput() const
        {
            return m_lastInput;
        }


        ActivationFunction Neuron::activationFunction() const
        {
            return m_activationFunction;
        }


        Neuron& Neuron::activationFunction(
                ActivationFunction activationFunction)
        {
            m_activationFunction = activationFunction;
            return *this;
        }


        double Neuron::activate(double sum)
        {
            m_lastInput = sum;
            m_lastResult = calculate(m_activationFunction, sum);
            return m_lastResult;
        }


        bool Neuron::operator ==(Neuron const& other) const
        {
            return (this->m_activationFunction == other.m_activationFunction
                    && this->m_lastInput == other.m_lastInput
                    && this->m_lastResult == other.m_lastResult);
        }


        bool Neuron::operator !=(const Neuron& other) const
        {
            return !(*this == other);
        }
    }
}
