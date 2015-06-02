#include <QVector>
#include <QMutableVectorIterator>

#include <cmath>

#include "Vector.h"

namespace Winzent {
    namespace ANN {


        Vector::Vector(): QVector<double>()
        {
        }


        Vector::Vector(int size): QVector<double>(size)
        {
        }


        Vector::Vector(const Vector &other): QVector<double>(other)
        {
        }


        Vector Vector::errors(const Vector &expected) const
        {
            Q_ASSERT(size() == expected.size());

            Vector errors(size());

            for (int i = 0; i != size(); ++i) {
                errors << at(i) - expected.at(i);
            }

            return errors;
        }


        double Vector::meanSquaredError(const Vector &expected) const
        {
            Q_ASSERT(size() == expected.size());
            Vector errorVector = errors(expected);

            double error = 0.0;
            int n = 0;

            for (; n != size(); ++n) {
                error += pow(errorVector.at(n), 2);
            }

            return error / static_cast<double>(n);
        }


        Vector &Vector::operator *(const int &x)
        {
            QMutableVectorIterator<double> i(*this);
            while (i.hasNext()) {
                i.next() *= x;
            }

            return *this;
        }


        Vector &Vector::operator =(const Vector &other)
        {
            if (*this == other) {
                return *this;
            }

            QVector<double>::operator =(other);
            return *this;
        }

    } // namespace ANN
} // namespace Winzent
