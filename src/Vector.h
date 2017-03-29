#ifndef WINZENT_ANN_VECTOR_H
#define WINZENT_ANN_VECTOR_H


#include <vector>
#include <ostream>


namespace Winzent {
    namespace ANN {


        //! \brief The real-valued number vector type
        typedef std::vector<double> Vector;


    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator <<(ostream &os, const Winzent::ANN::Vector &vector);
}


#endif // WINZENT_ANN_VECTOR_H
