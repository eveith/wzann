#include <cmath>
#include <limits>
#include <cstddef>
#include <cassert>
#include <ostream>

#include "Vector.h"
#include "TrainingItem.h"

#include "TrainingSet.h"


namespace Winzent {
    namespace ANN {

        TrainingSet::TrainingSet():
                m_targetError(0),
                m_maxNumEpochs(std::numeric_limits<size_t>::max()),
                m_error(std::numeric_limits<double>::max())
        {
        }


        TrainingSet::TrainingSet(
                TrainingItems trainingData,
                double targetError,
                size_t maxNumEpochs):
                    trainingItems(trainingData),
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(std::numeric_limits<double>::max())
        {
        }


        TrainingSet::TrainingSet(TrainingSet const& other):
                trainingItems(other.trainingItems),
                m_targetError(other.m_targetError),
                m_maxNumEpochs(other.m_maxNumEpochs),
                m_epochs(other.m_epochs),
                m_error(other.m_error)
        {
        }


        double TrainingSet::targetError() const
        {
            return m_targetError;
        }


        TrainingSet& TrainingSet::targetError(double targetError)
        {
            m_targetError = targetError;
            return *this;
        }


        double TrainingSet::error() const
        {
            return m_error;
        }


        size_t TrainingSet::maxEpochs() const
        {
            return m_maxNumEpochs;
        }


        TrainingSet &TrainingSet::maxEpochs(size_t maxEpochs)
        {
            m_maxNumEpochs = maxEpochs;
            return *this;
        }


        size_t TrainingSet::epochs() const
        {
            return m_epochs;
        }


        TrainingSet &TrainingSet::operator <<(TrainingItem&& item)
        {
            trainingItems.push_back(item);
            return *this;
        }


        void TrainingSet::push_back(TrainingItem&& item)
        {
            trainingItems.push_back(item);
        }


        void TrainingSet::push_back(TrainingSet const& trainingSet)
        {
            for (auto const& i: trainingSet.trainingItems) {
                push_back(TrainingItem(i));
            }
        }


        TrainingSet &TrainingSet::operator =(TrainingSet const& rhs)
        {
            if (this == &rhs) {
                return *this;
            }

            this->trainingItems = rhs.trainingItems;

            this->m_epochs      = rhs.m_epochs;
            this->m_maxNumEpochs= rhs.m_maxNumEpochs;

            this->m_error       = rhs.m_error;
            this->m_targetError = rhs.m_targetError;

            return *this;
        }
    }
}


namespace std {
    ostream& operator <<(
            ostream& os,
            Winzent::ANN::TrainingSet::TrainingItems const& trainingData)
    {
        os << "TrainingData = (";
        for (auto const& i: trainingData) {
            os << i;
            if (&i != &(trainingData.back())) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }


    ostream& operator <<(
            ostream& os,
            Winzent::ANN::TrainingSet const& trainingSet)
    {
        os
                << "TrainingSet = ("
                << "TargetError = " << trainingSet.targetError()
                << ", Error = " << trainingSet.error()
                << ", MaxEpochs = " << trainingSet.maxEpochs()
                << ", epochs = " << trainingSet.epochs()
                << ", " << trainingSet.trainingItems
                << ")";
        return os;
    }
} // namespace std


constexpr const char Winzent::ANN::JsonSchema<Winzent::ANN::TrainingSet>
        ::schemaURI[];
