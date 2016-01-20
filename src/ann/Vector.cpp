#include "Vector.h"

namespace Winzent {
    namespace ANN {
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator <<(ostream &os, const Winzent::ANN::Vector &vector)
    {
        os << "(";

        for (auto it = vector.begin(); it != vector.end(); it++) {
            os << *it;
            if (it != vector.end()-1) {
                os << ", ";
            }
        }

        os << ")";
        return os;
    }

}
