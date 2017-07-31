#include <cmath>
#include <cstddef>
#include <cassert>
#include <ostream>

#include "Vector.h"
#include "TrainingItem.h"

namespace wzann {
    TrainingItem::TrainingItem()
    {
    }


    TrainingItem::TrainingItem(
            Vector const& input,
            Vector const& expectedOutput):
                m_input(input),
                m_expectedOutput(expectedOutput)
    {
    }


    TrainingItem::TrainingItem(Vector const& input):
            TrainingItem(input, Vector())
    {
    }


    TrainingItem::TrainingItem(TrainingItem const& rhs):
            m_input(rhs.m_input),
            m_expectedOutput(rhs.m_expectedOutput)
    {
    }


    TrainingItem::TrainingItem(TrainingItem&& rhs):
            m_input(std::move(rhs.m_input)),
            m_expectedOutput(std::move(rhs.m_expectedOutput))
    {
    }


    Vector TrainingItem::input() const
    {
        return m_input;
    }


    Vector TrainingItem::expectedOutput() const
    {
        return m_expectedOutput;
    }


    bool TrainingItem::outputRelevant() const
    {
        return m_expectedOutput.size() > 0;
    }


    TrainingItem& TrainingItem::operator =(TrainingItem const& rhs)
    {
        if (this == &rhs) {
            return *this;
        }

        this->m_input = rhs.m_input;
        this->m_expectedOutput = rhs.m_expectedOutput;

        return *this;
    }
} // namespace wzann


namespace std {
    ostream& operator <<(
            ostream& os,
            wzann::TrainingItem const& trainingItem)
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
} // namespace std
