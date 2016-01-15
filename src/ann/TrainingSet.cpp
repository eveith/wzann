#include <cmath>
#include <limits>
#include <cstddef>
#include <ostream>

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "JsonSerializable.h"
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


        void TrainingItem::clear()
        {
            m_input.clear();
            m_expectedOutput.clear();
            m_outputRelevant = false;
        }


        void TrainingItem::fromJSON(const QJsonDocument &json)
        {
            clear();
            QJsonObject o = json.object();

            m_input = from_json(
                    QJsonDocument(o["input"].toArray()));
            m_expectedOutput = from_json(
                    QJsonDocument(o["expectedOutput"].toArray()));
            m_outputRelevant = (m_expectedOutput.size() > 0);
        }


        QJsonDocument TrainingItem::toJSON() const
        {
            QJsonObject o;

            o["input"] = to_json(m_input).array();
            o["expectedOutput"] = to_json(m_expectedOutput).array();

            return QJsonDocument(o);
        }


        TrainingSet::TrainingSet():
                m_targetError(0),
                m_maxNumEpochs(std::numeric_limits<size_t>::max()),
                m_error(std::numeric_limits<qreal>::max())
        {
        }


        TrainingSet::TrainingSet(
                TrainingItems trainingData,
                const qreal &targetError,
                const size_t &maxNumEpochs):
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(std::numeric_limits<qreal>::max())
        {
            for (const auto &i: trainingData) {
                this->trainingData.push_back(TrainingItem(i));
            }
        }


        qreal TrainingSet::targetError() const
        {
            return m_targetError;
        }


        TrainingSet &TrainingSet::targetError(const qreal &targetError)
        {
            m_targetError = targetError;
            return *this;
        }


        qreal TrainingSet::error() const
        {
            return m_error;
        }


        size_t TrainingSet::maxEpochs() const
        {
            return m_maxNumEpochs;
        }


        TrainingSet &TrainingSet::maxEpochs(const size_t &maxEpochs)
        {
            m_maxNumEpochs = maxEpochs;
            return *this;
        }


        size_t TrainingSet::epochs() const
        {
            return m_epochs;
        }


        TrainingSet &TrainingSet::operator <<(const TrainingItem &item)
        {
            trainingData.push_back(item);
            return *this;
        }


        void TrainingSet::clear()
        {
            trainingData.clear();
            m_maxNumEpochs = 0;
            m_epochs = 0;
            m_targetError = 0.0;
            m_error = std::numeric_limits<qreal>::max();
        }


        void TrainingSet::fromJSON(const QJsonDocument &json)
        {
            clear();
            QJsonObject o = json.object();

            m_epochs = static_cast<size_t>(o["epochs"].toInt());
            m_maxNumEpochs = static_cast<size_t>(o["maxEpochs"].toInt());
            m_error = o["error"].toDouble();
            m_targetError = o["targetError"].toDouble();

            QJsonArray jsonTrainingItems = o["trainingItems"].toArray();
            for (const auto &i: jsonTrainingItems) {
                TrainingItem ti;
                ti.fromJSON(QJsonDocument(i.toObject()));
                trainingData.push_back(ti);
            }
        }


        QJsonDocument TrainingSet::toJSON() const
        {
            QJsonObject o;

            o["epochs"] = static_cast<int>(epochs());
            o["maxEpochs"] = static_cast<int>(maxEpochs());
            o["error"] = error();
            o["targetError"] = targetError();

            QJsonArray jsonTrainingItems;
            for (const auto &i: trainingData) {
                jsonTrainingItems.push_back(i.toJSON().object());
            }
            o["trainingItems"] = jsonTrainingItems;

            return QJsonDocument(o);
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
                << ", " << trainingSet.trainingData
                << ")";
        return os;
    }
}


constexpr const char Winzent::JsonSchema<Winzent::ANN::TrainingSet>
        ::schemaURI[];
