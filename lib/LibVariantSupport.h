#ifndef WZANN_LIBVARIANTSUPPORT_H_
#define WZANN_LIBVARIANTSUPPORT_H_


#include <Variant/Variant.h>


namespace wzann {


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
     * \brief Heap-allocates a new object of type T from a variant.
     *
     * This general function is actually a stub: Classes that wish to
     * be available for deserialization from a libvariant::Variant must
     * specialize this function template.
     *
     * \sa to_variant()
     */
    template <typename T>
    T* new_from_variant(libvariant::Variant const&);


    /*!
     * \brief Creates a new object of the type T on the stack         *
     *
     * This general function is actually a stub: Classes that wish to
     * be available for deserialization from a libvariant::Variant must
     * specialize this function template.
     *
     * \sa to_variant()
     */
    template <typename T>
    T from_variant(libvariant::Variant const&);
} // namespace wzann


#endif // WZANN_LIBVARIANTSUPPORT_H_
