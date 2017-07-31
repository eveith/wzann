#include "Vector.h"


namespace std {
    ostream &operator <<(ostream& os, wzann::Vector const& vector)
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
