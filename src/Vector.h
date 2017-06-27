#ifndef WINZENT_ANN_VECTOR_H
#define WINZENT_ANN_VECTOR_H


#include <vector>
#include <ostream>

#include "LibVariantSupport.h"


namespace Winzent {
    namespace ANN {


        //! \brief The real-valued number vector type
        typedef std::vector<double> Vector;


        template <> inline libvariant::Variant
        to_variant(Vector const& v)
        {
            return libvariant::Variant(v);
        }


        template <> inline Vector
        from_variant(libvariant::Variant const& variant)
        {
            Vector v;
            auto const& list = variant.AsList();

            for (auto const& val : list) {
                v.push_back(val.AsDouble());
            }

            return v;
        }
    } // namespace ANN
} // namespace Winzent


namespace std {
    ostream &operator <<(ostream &os, const Winzent::ANN::Vector &vector);
}


#endif // WINZENT_ANN_VECTOR_H
