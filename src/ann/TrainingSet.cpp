#include <cmath>
#include <limits>
#include <cstddef>
#include <ostream>

#include "QtContainerOutput.h"

#include "Exception.h"
#include "TrainingSet.h"


namespace Winzent {
    namespace ANN {


        TrainingItem::TrainingItem(
                const Vector &input,
                const Vector &expectedOutput):
                    m_input(input),
                    m_expectedOutput(expectedOutput),
                    m_outputRelevant(true)
        {
        }


        TrainingItem::TrainingItem(const Vector &input):
                TrainingItem(input, Vector())
        {
            m_outputRelevant = false;
        }


        TrainingItem::TrainingItem(const TrainingItem &rhs):
                TrainingItem(rhs.m_input, rhs.m_expectedOutput)
        {
            m_outputRelevant = rhs.outputRelevant();
        }


        const Vector TrainingItem::input() const
        {
            return m_input;
        }


        const Vector TrainingItem::expectedOutput() const
        {
            return m_expectedOutput;
        }


        bool TrainingItem::outputRelevant() const
        {
            return m_outputRelevant;
        }


        Vector TrainingItem::error(const Vector &actualOutput) const
                throw(LayerSizeMismatchException)
        {
            if (actualOutput.size() == expectedOutput().size()) {
                throw LayerSizeMismatchException(
                        actualOutput.size(),
                        expectedOutput().size());
            }

            Vector r(expectedOutput().size());

            for (auto i = 0; i != actualOutput.size(); ++i) {
                r[i] = expectedOutput().at(i) - actualOutput.at(i);
            }

            return r;
        }


        TrainingSet::TrainingSet(
                TrainingItems trainingData,
                const qreal &targetError,
                const size_t &maxNumEpochs):
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(std::numeric_limits<qreal>::infinity())
        {
            for (const auto &i: trainingData) {
                m_trainingData.push_back(TrainingItem(i));
            }
        }


        qreal TrainingSet::targetError() const
        {
            return m_targetError;
        }


        qreal TrainingSet::error() const
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


        TrainingSet::TrainingItems TrainingSet::trainingData() const
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


namespace std {
    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingItem &trainingItem)
    {
        os
                << "TrainingItem = ("
                << "Input = "
                << trainingItem.input()
                << ", ExpectedOutput = "
                << trainingItem.expectedOutput()
                << ", OutputRelevant = "
                << trainingItem.outputRelevant()
                << ")";
        return os;
    }


    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingSet::TrainingItems &trainingData)
    {
        os << "TrainingData = (";
        for (const auto &i: trainingData) {
            os << i;
            if (&i != &(trainingData.back())) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }


    ostream &operator <<(
            ostream &os,
            const Winzent::ANN::TrainingSet &trainingSet)
    {
        os
                << "TrainingSet = ("
                << "TargetError = " << trainingSet.targetError()
                << ", Error = " << trainingSet.error()
                << ", MaxEpochs = " << trainingSet.maxEpochs()
                << ", epochs = " << trainingSet.epochs()
                << ", " << trainingSet.trainingData()
                << ")";
        return os;
    }
}
