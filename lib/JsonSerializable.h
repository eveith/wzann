#ifndef WZANN_JSONSERIALIZABLE_H_
#define WZANN_JSONSERIALIZABLE_H_


#include <string>
#include <streambuf>

#include <Variant/Schema.h>
#include <Variant/Variant.h>

#include <boost/static_assert.hpp>

#include "LibVariantSupport.h"
#include "SchemaValidationException.h"


namespace wzann {


    /*!
     * \brief Schema URI strict for validation
     *
     * Classes that are serializable and have a JSON schema attached need
     * to provide a template specialization of this struct. Doing so
     * enables the automatic schema checking.
     *
     * The specialized template struct must have a `schemaURI` constant.
     */
    template <class C>
    struct JsonSchema
    {
        static constexpr char const schemaURI[] = "";
    };


    template <
        class C,
        typename std::enable_if<(
            std::extent<decltype(wzann::JsonSchema<C>::schemaURI)>
                ::value <= 1),
            int>::type = 0>
    inline C from_json(std::string& json)
    {
        auto variant(libvariant::DeserializeJSON(json));
        return from_variant<C>(variant);
    }


    template <
        class C,
        typename std::enable_if<(
            std::extent<decltype(wzann::JsonSchema<C>::schemaURI)>
                ::value > 1),
            int>::type = 0>
    inline C from_json(
            std::string& json,
            char const schemaURI[] = wzann::JsonSchema<C>::schemaURI)
    {
        libvariant::Variant jsonSchema(libvariant::DeserializeJSONFile(
                schemaURI));
        libvariant::Variant jsonData = libvariant::DeserializeJSON(
                json);
        auto r(libvariant::SchemaValidate(
                jsonSchema,
                jsonData));

        if (r.Error()) {
            throw SchemaValidationException(r);
        }

        return from_variant<C>(jsonData);
    }


    template <
        class C,
        typename std::enable_if<(
            std::extent<decltype(wzann::JsonSchema<C>::schemaURI)>
                ::value <= 1),
            int>::type = 0>
    inline C* new_from_json(std::string& json)
    {
        auto variant(libvariant::DeserializeJSON(json));
        return new_from_variant<C>(variant);
    }


    template <
        class C,
        typename std::enable_if<(
            std::extent<decltype(wzann::JsonSchema<C>::schemaURI)>
                ::value > 1),
            int>::type = 0>
    inline C* new_from_json(std::string& json)
    {
        auto jsonSchema(libvariant::DeserializeJSONFile(
                wzann::JsonSchema<C>::schemaURI));
        auto jsonData(libvariant::DeserializeJSON(json));
        auto r(libvariant::SchemaValidate(jsonSchema, jsonData));

        if (r.Error()) {
            throw SchemaValidationException(r);
            return nullptr;
        }

        return new_from_variant<C>(jsonData);
    }


    template <class C>
    inline std::string to_json(C const& serializable)
    {
        return libvariant::SerializeJSON(to_variant<C>(serializable));
    }
} // namespace wzann

#endif // WZANN_JSONSERIALIZABLE_H_
