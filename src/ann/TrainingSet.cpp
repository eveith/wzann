/*
 * TrainingSet.cpp

 *
 *  Created on: 18.10.2012
 *      Author: eveith
 */


#include <cmath>

#include "Exception.h"
#include "TrainingSet.h"


namespace Winzent {
    namespace ANN {


        TrainingItem::TrainingItem(
                const ValueVector &input,
                const ValueVector &expectedOutput):
                    m_input(),
                    m_expectedOutput(),
                    m_outputRelevant(true)
        {
            // Copy all values so that we don't lose it and can modify them as
            // we wish:

            foreach (qreal i, input) {
                m_input.append(qreal(i));
            }

            foreach (qreal i, expectedOutput) {
                m_expectedOutput.append(qreal(i));
            }
        }


        TrainingItem::TrainingItem(const ValueVector &input):
                TrainingItem(input, ValueVector())
        {
            m_outputRelevant = false;
        }


        TrainingItem::TrainingItem(const TrainingItem &rhs):
                TrainingItem(rhs.m_input, rhs.m_expectedOutput)
        {
            m_outputRelevant = rhs.outputRelevant();
        }


        const ValueVector TrainingItem::input() const
        {
            return m_input;
        }


        const ValueVector TrainingItem::expectedOutput() const
        {
            return m_expectedOutput;
        }


        bool TrainingItem::outputRelevant() const
        {
            return m_outputRelevant;
        }


        ValueVector TrainingItem::error(const ValueVector &actualOutput) const
                throw(LayerSizeMismatchException)
        {
            if (actualOutput.size() == expectedOutput().size()) {
                throw LayerSizeMismatchException(
                        actualOutput.size(),
                        expectedOutput().size());
            }

            ValueVector r(expectedOutput().size());

            for (int i = 0; i != actualOutput.size(); ++i) {
                r[i] = expectedOutput().at(i) - actualOutput.at(i);
            }

            return r;
        }


        TrainingSet::TrainingSet(
                QList<TrainingItem> trainingData,
                double targetError,
                int maxNumEpochs):
                    m_trainingData(QList<TrainingItem>()),
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(INFINITY)
        {
            foreach (TrainingItem i, trainingData) {
                m_trainingData << TrainingItem(i);
            }
        }


        TrainingSet::~TrainingSet()
        {
        }


        double TrainingSet::targetError() const
        {
            return m_targetError;
        }


        double TrainingSet::error() const
        {
            return m_error;
        }


        int TrainingSet::maxEpochs() const
        {
            return m_maxNumEpochs;
        }


        int TrainingSet::epochs() const
        {
            return m_epochs;
        }


        const QList<TrainingItem> &TrainingSet::trainingData() const
        {
            return m_trainingData;
        }


        TrainingSet &TrainingSet::operator<<(const TrainingItem &item)
        {
            m_trainingData << item;
            return *this;
        }


        TrainingItem &TrainingSet::operator [](const int &index)
        {
            return m_trainingData[index];
        }
    }
}
