#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H


#include <Variant/Variant.h>


namespace Winzent {
    namespace ANN {


        /*!
         * \brief Converts an object to its Variant representation
         *
         * This general function is actually a stub: Classes that wish to
         * be available for serialization to variant must specialize this
         * function template.
         *
         * \sa from_variant()
         */
        template <typename T>
        libvariant::Variant to_variant(T const&);


        /*!
         * \brief Converts an object to its Variant representation
         *
         * This general function is actually a stub: Classes that wish to
         * be available for deserialization from a libvariant::Variant must
         * specialize this function template.
         *
         * \sa to_variant()
         */
        template <typename T>
        T* new_from_variant(libvariant::Variant const&);
    }
}


#endif // SERIALIZABLE_H
