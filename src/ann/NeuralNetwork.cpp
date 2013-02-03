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



        Layer::Layer(QObject *parent):
                QObject(parent),
                neurons(QList<Neuron*>())
        {
        }


        int Layer::size() const
        {
            return neurons.size();
        }


        Layer* Layer::clone() const
        {
            Layer* layerClone = new Layer();

            foreach (Neuron* n, neurons) {
                Neuron* neuronClone = n->clone();
                neuronClone->setParent(layerClone);
                layerClone->neurons << neuronClone;
            }

            return layerClone;
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
                index += m_layers[i]->size();
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


        Neuron* NeuralNetwork::neuron(const int &index)
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

            // Grow weight matrix:

            int weightMatrixTargetSize = m_weightMatrix.size() + layer->size();

            for (int i = 0; i != layer->size(); ++i) {
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

            return *this;
        }


        Layer*& NeuralNetwork::operator[](const int &i)
        {
            return m_layers[i];
        }


        void NeuralNetwork::inputLayer(Layer *layer)
        {
            m_layers.prepend(layer);

            // TODO: Layer/weight/stuff modification!
        }


        Layer* NeuralNetwork::inputLayer() const
        {
            return m_layers.first();
        }


        void NeuralNetwork::outputLayer(Layer *layer)
        {
            *this << layer;
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

            ValueVector output(layer->size());

            for (int i = 0; i != input.size(); ++i) {
                output[i] = layer->neurons.at(i)->activate(input.at(i));
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

            ValueVector output(toLayerSize);

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
            if (input.size() != m_layers[0]->size()) {
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
