/*
 * TrainingSet.cpp

 *
 *  Created on: 18.10.2012
 *      Author: eveith
 */


#include <cmath>
#include <limits>
#include <cstddef>

#include "Exception.h"
#include "TrainingSet.h"


namespace Winzent {
    namespace ANN {


        TrainingItem::TrainingItem(
                const ValueVector &input,
                const ValueVector &expectedOutput):
                    m_input(input),
                    m_expectedOutput(expectedOutput),
                    m_outputRelevant(true)
        {
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

            for (auto i = 0; i != actualOutput.size(); ++i) {
                r[i] = expectedOutput().at(i) - actualOutput.at(i);
            }

            return r;
        }


        TrainingSet::TrainingSet(
                QList<TrainingItem> trainingData,
                const double &targetError,
                const size_t &maxNumEpochs):
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(std::numeric_limits<double>::infinity())
        {
            for (const auto &i: trainingData) {
                m_trainingData.push_back(TrainingItem(i));
            }
        }


        double TrainingSet::targetError() const
        {
            return m_targetError;
        }


        double TrainingSet::error() const
        {
            return m_error;
        }


        size_t TrainingSet::maxEpochs() const
        {
            return m_maxNumEpochs;
        }


        size_t TrainingSet::epochs() const
        {
            return m_epochs;
        }


        const QList<TrainingItem> &TrainingSet::trainingData() const
        {
            return m_trainingData;
        }


        TrainingSet &TrainingSet::operator <<(const TrainingItem &item)
        {
            m_trainingData.push_back(item);
            return *this;
        }


        TrainingItem &TrainingSet::operator [](const size_t &index)
        {
            return m_trainingData[index];
        }
    }
}
