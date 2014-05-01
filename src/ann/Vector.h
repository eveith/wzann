#ifndef WINZENT_ANN_VECTOR_H
#define WINZENT_ANN_VECTOR_H

#include <QVector>


namespace Winzent {
    namespace ANN {

        class Vector: public QVector<qreal>
        {
        public:


            /*!
             * \brief Creates a new, empty vector.
             */
            Vector();


            /*!
             * \brief Creates a new vector and allocates space for
             *
             * This allows run-time optimization of the Vector. It does not set
             * an upper limit; instead, it allows to reserve memory for a
             * certain amount of elements beforehand in order to avoid
             * reallocations.
             *
             * \param[in] size The number of elements to reserve space for
             */
            explicit Vector(int size);


            /*!
             * \brief Copy constructor
             *
             * \param[in] other The Vector of which to create a copy
             */
            Vector(const Vector &other);


            /*!
             * \brief Calculates the mean squared error given a Vector of true
             *  values.
             *
             * This method considers the current vector to be the predicted
             * values and the parameter vector to contain the true values, and
             * then calculates $\frac{1}{n} \sum_0^{i} (expected_i - this_i)^2$.
             *
             * \param[in] expected The true values
             *
             * \return The MSE
             */
            qreal meanSquaredError(const Vector &expected) const;


            /*!
             * \brief Computes a vector of errors given another Vector holding
             *  the true values
             *
             * This method computes $e_i = this_i - expected_i$ for all items
             * of the vector `expected`.
             *
             * \param[in] expected The true values
             *
             * \return The error
             */
            Vector errors(const Vector &expected) const;


            /*!
             * \brief Multiplies each member of this Vector with a constant.
             *
             * \param[in] x The constant
             *
             * \return `*this`
             */
            Vector &operator*(const int &x);


            /*!
             * \brief Assignment operator
             */
            Vector &operator=(const Vector &other);
        };

    } // namespace ANN
} // namespace Winzent

#endif // WINZENT_ANN_VECTOR_H
