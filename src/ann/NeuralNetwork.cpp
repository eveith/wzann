/*!
 * \file	NeuralNetwork.cpp
 * \brief
 * \date	17.12.2012
 * \author	eveith
 */


#include <QtDebug>
#include <QObject>
#include <QList>
#include <QVector>
#include <QByteArray>
#include <QTextStream>

#include <qjson/serializer.h>
#include <qjson/qobjecthelper.h>

#include "Layer.h"
#include "Neuron.h"
#include "Exception.h"
#include "NeuralNetworkPattern.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "TrainingAlgorithm.h"


namespace Winzent
{
    namespace ANN
    {

        const char NeuralNetwork::VERSION[] = "1.0";


        Weight::Weight(QObject *parent):
            QObject(parent),
            value(0.0),
            fixed(false)
        {
        }


        Weight* Weight::clone() const
        {
            Weight* clone = new Weight();

            clone->value = value;
            clone->fixed = fixed;

            return clone;
        }


        double Weight::weight() const
        {
            return value;
        }


        void Weight::weight(double weight) throw(WeightFixedException)
        {
            if (fixed) {
                throw WeightFixedException();
            } else {
                value = weight;
            }
        }


        void Weight::setRandomWeight(const double &min, const double &max)
                throw(WeightFixedException)
        {
            weight(min + (qrand() * abs(max-min)
                    / static_cast<double>(RAND_MAX)));
        }


        Weight::operator double() const
        {
            return value;
        }


        double Weight::operator*(const double& rhs) const
        {
            return value * rhs;
        }


        NeuralNetwork::NeuralNetwork(QObject* parent):
                QObject(parent),
                m_layers(QList<Layer*>()),
                m_weightMatrix(WeightMatrix()),
                m_pattern(NULL)
        {
        }


        NeuralNetwork::NeuralNetwork(const NeuralNetwork& rhs):
                QObject(rhs.parent()),
                m_layers(QList<Layer*>(rhs.m_layers)),
                m_weightMatrix(WeightMatrix()),
                m_pattern(NULL)
        {
            // Clone layers:

            foreach(Layer *l, rhs.m_layers) {
                Layer *layerClone = l->clone();
                layerClone->setParent(this);
                m_layers << layerClone;
            }

            // Clone weights, take care of unconnected neurons (weight == NULL):

            foreach (QList<Weight*> i, rhs.m_weightMatrix) {
                QList<Weight*> list;

                foreach (Weight *w, i) {
                    if (NULL ==  w) { // Not connected
                        list << NULL;
                    } else {
                        list << w->clone();
                    }
                }

                m_weightMatrix << list;
            }

            // Make sure the cloned pattern has the correct parent object:

            if (NULL != rhs.m_pattern) {
                m_pattern = rhs.m_pattern->clone();
                m_pattern->setParent(this);
            }
        }


        NeuralNetwork::~NeuralNetwork()
        {
        }


        bool NeuralNetwork::containsNeuron(const Neuron *neuron) const
        {
            if (NULL == neuron) {
                return false;
            }

            foreach (Layer *l, m_layers) {
                foreach (Neuron *n, l->neurons) {
                    if (neuron == n) {
                        return true;
                    }
                }
            }

            return false;
        }


        int NeuralNetwork::findNeuron(const Neuron *neuron) const
        {
            int index = 0;

            foreach (Layer *l, m_layers) {
                foreach (Neuron *n, l->neurons) {
                    if (neuron == n) {
                        return index;
                    }
                    ++index;
                }
            }

            return -1;
        }


        int NeuralNetwork::translateIndex(
                const int &layer,
                const int &neuronIndex) const
        {
            if (layer >= m_layers.size()) {
                return -1;
            }

            int index = 0;

            for (int i = 0; i != layer; ++i) {
                index += m_layers[i]->neurons.size();
            }

            index += neuronIndex;
            return index;
        }


        bool NeuralNetwork::neuronConnectionExists(
                const int &i,
                const int &j) const
        {
            bool b = false;

            if (i < m_weightMatrix.size() && j < m_weightMatrix.at(i).size()
                    && NULL != m_weightMatrix.at(i).at(j)) {
                b = true;
            }

            return b;
        }


        Weight* NeuralNetwork::weight(const int &i, const int &j) const
                throw(NoConnectionException)
        {
            if (!neuronConnectionExists(i, j)) {
                throw NoConnectionException();
            }

            return m_weightMatrix.at(i).at(j);
        }


        void NeuralNetwork::weight(const int &i, const int &j, double value)
                throw(NoConnectionException)
        {
            if (! neuronConnectionExists(i, j)) {
                throw NoConnectionException();
            }
            m_weightMatrix[i][j]->value = value;
        }


        Neuron* NeuralNetwork::neuron(const int &index) const
        {
            int i = 0;

            foreach (Layer *l, m_layers) {
                foreach (Neuron *n, l->neurons) {
                    if (index == i) {
                        return n;
                    }
                    i++;
                }
            }

            return NULL;
        }


        QHash<Neuron*, Weight*> NeuralNetwork::connectedNeurons(
                const int &index) const
        {
            Q_ASSERT(index > 0);
            Q_ASSERT(index < m_weightMatrix.size());

            QHash<Neuron*, Weight*> ret;

            for (int i = 0; i != m_weightMatrix[index].size(); ++i) {
                if (!neuronConnectionExists(index, i)) {
                    continue;
                }

                ret.insert(neuron(i), m_weightMatrix[index][i]);
            }

            return ret;
        }


        QHash<Neuron*, Weight*> NeuralNetwork::connectedNeurons(
                const Neuron *neuron) const
        {
            return connectedNeurons(findNeuron(neuron));
        }

        void NeuralNetwork::connectNeurons(const int &i, const int &j)
        {
            Q_ASSERT(m_weightMatrix.size() >= i);
            Q_ASSERT(m_weightMatrix.size() == m_weightMatrix[i].size());
            Q_ASSERT(m_weightMatrix[i].size() >= j);

            if (! neuronConnectionExists(i, j)) {
                m_weightMatrix[i][j] = new Weight();
            }
        }


        void NeuralNetwork::connectNeurons(Neuron *i, Neuron *j)
        {
            connectNeurons(findNeuron(i), findNeuron(j));
        }


        NeuralNetwork& NeuralNetwork::operator<<(Layer *layer)
        {
            m_layers << layer;
            layer->setParent(this);

            // Grow weight matrix; include the bias neuron in the calculation:

            int newLayerSize = layer->neurons.size();
            int weightMatrixTargetSize = m_weightMatrix.size() + newLayerSize;

            for (int i = 0; i != newLayerSize; ++i) {
                QList<Weight*> list;
                m_weightMatrix.append(list);
            }

            // Adjust row sizes:

            for (int i = 0; i != m_weightMatrix.size(); ++i) {
                if (m_weightMatrix[i].size() < weightMatrixTargetSize) {
                    int size = m_weightMatrix[i].size();
                    for (int j = 0; j != weightMatrixTargetSize - size; ++j) {
                        (m_weightMatrix[i]) << NULL;
                    }
                }

                Q_ASSERT(m_weightMatrix[i].size() == m_weightMatrix.size());
            }

            // Make sure the bias connection exists:

            int biasIndex = findNeuron(layer->biasNeuron());

            for (int i = 0; i != layer->size(); ++i) {
                int neuronIndex = biasIndex - layer->size() + i;
                connectNeurons(biasIndex, neuronIndex);
                weight(biasIndex, neuronIndex, 1.0);
            }

            return *this;
        }


        Layer*& NeuralNetwork::operator[](const int &i)
        {
            return m_layers[i];
        }


        Layer* NeuralNetwork::inputLayer() const
        {
            return m_layers.first();
        }


        Layer* NeuralNetwork::outputLayer() const
        {
            return m_layers.last();
        }


        void NeuralNetwork::configure(const NeuralNetworkPattern *pattern)
        {
            // Get rid of the old pattern, if one exists:

            if (NULL != m_pattern) {
                delete m_pattern;
            }

            m_pattern = pattern->clone();
            m_pattern->setParent(this);

            m_pattern->configureNetwork(this);
        }


        void NeuralNetwork::train(
                TrainingAlgorithm *trainingStrategy,
                TrainingSet *trainingSet)
        {
            trainingStrategy->train(this, trainingSet);
        }


        ValueVector NeuralNetwork::calculateLayer(
                Layer *layer,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            if (layer->size() != input.size()) {
                throw LayerSizeMismatchException(input.size(), layer->size());
            }

            ValueVector output;
            output.fill(0.0, layer->size());

            int biasIndex = findNeuron(layer->biasNeuron());

            for (int i = 0; i != input.size(); ++i) {
                double sum = input.at(i);

                // Add bias neuron. We ignore the bias neuron in the input layer
                // even when it's there; it does not make sense to include the
                // bias neuron in the input layer since its output would be
                // overwritten by the input anyways.

                if(m_layers.first() == layer) {
                    continue;
                }

                int neuronIndex = findNeuron((*layer)[i]);
                if (neuronConnectionExists(biasIndex, neuronIndex)) {
                    sum += layer->biasNeuron()->activate(1.0)
                            * weight(biasIndex, neuronIndex)->value;
                }

                output[i] = (*layer)[i]->activate(sum);
            }

            return output;
        }


        ValueVector NeuralNetwork::calculateLayer(
                const int &layerIndex,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            return this->calculateLayer((*this)[layerIndex], input);
        }


        ValueVector NeuralNetwork::calculateLayerTransition(
                const int &fromLayer,
                const int &toLayer,
                const ValueVector &input)
                    throw(LayerSizeMismatchException)
        {
            int fromLayerSize   = (*this)[fromLayer]->size();
            int toLayerSize     = (*this)[toLayer]->size();

            if (input.size() != fromLayerSize) {
                throw LayerSizeMismatchException(input.size(), fromLayerSize);
            }

            ValueVector output;
            output.fill(0.0, toLayerSize);
            Q_ASSERT(output.size() == toLayerSize);

            for (int i = 0; i != fromLayerSize; ++i) {
                int fromNeuronIndex = translateIndex(fromLayer, i);

                for (int j = 0; j != toLayerSize; ++j) {
                    int toNeuronIndex = translateIndex(toLayer, j);

                    if (!neuronConnectionExists(
                            fromNeuronIndex,
                            toNeuronIndex)) {
                        continue;
                    }

                    output[j] += *(weight(fromNeuronIndex, toNeuronIndex))
                            * input.at(i);
                }
            }

            return output;
        }


        ValueVector NeuralNetwork::calculate(const ValueVector &input)
                throw(LayerSizeMismatchException)
        {
            if (input.size() != m_layers.first()->size()) {
                throw LayerSizeMismatchException(
                        m_layers.first()->size(),
                        input.size());
            }

            return m_pattern->calculate(this, input);
        }


        QTextStream& operator<<(QTextStream &out, const NeuralNetwork &network)
        {
            QVariantMap outList;

            outList.insert("Version", NeuralNetwork::VERSION);

            QList<QVariant> layersList;

            outList.insert("Layers", layersList);

            for (int i = 0; i != network.m_layers.size(); ++i) {
                QVariantMap layerMap;
                QVariantList neuronsList;

                QList<Neuron*> neurons = network.m_layers[i]->neurons;
                for (int j = 0; j != neurons.size(); ++j) {
                    QVariantMap neuronMap;

                    neuronMap.insert(
                            "ActivationFunction",
                            neurons[j]->m_activationFunction
                                ->metaObject()->className());
                    neuronMap.insert(
                            "LastResult",
                            neurons[j]->lastResult());

                    neuronsList.append(neuronMap);
                }

                layerMap.insert("Neurons", neuronsList);
                layersList.append(layerMap);
            }

            outList.insert("Layers", layersList);

            QVariantMap weightMatrix;

            for (int i = 0; i != network.m_weightMatrix.size(); ++i) {
                QVariantList row;
                QList<Weight*> weights = network.m_weightMatrix.at(i);

                for (int j = 0; j != weights.size(); ++j) {
                    Weight* w = weights.at(j);
                    if (NULL == w) {
                        row.append(0);
                        continue;
                    }
                    QVariantMap weight;

                    weight.insert("Value", w->value);
                    weight.insert("Fixed", w->fixed);

                    row.append(weight);
                }

                weightMatrix.insert(QString::number(i), row);
            }

            outList.insert("WeightMatrix", weightMatrix);

            // Serialize:

            QJson::Serializer serializer;
            serializer.setIndentMode(QJson::IndentFull);

            bool ok;
            QByteArray json = serializer.serialize(outList, &ok);

            if (!ok) {
                qCritical()
                        << "Error during ANN serialization: "
                        << serializer.errorMessage();
                return out;
            }

            out << json;
            return out;
        }
    }
}
