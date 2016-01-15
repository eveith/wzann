#ifndef WINZENT_ANN_VECTOR_H
#define WINZENT_ANN_VECTOR_H

#include <QVector>
#include <QJsonArray>
#include <QJsonDocument>


namespace Winzent {
    namespace ANN {


        typedef QVector<qreal> Vector;


        inline QJsonDocument to_json(const Vector &vector)
        {
            QJsonArray jsonArray;

            std::copy(
                    vector.begin(),
                    vector.end(),
                    std::back_inserter(jsonArray));

            return QJsonDocument(jsonArray);
        }


        inline Vector from_json(const QJsonDocument &jsonDocument)
        {
            QJsonArray jsonArray = jsonDocument.array();
            Vector vector;

            for (const auto &i: jsonArray) {
                vector.push_back(i.toDouble());
            }

            return vector;
        }
    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_VECTOR_H
