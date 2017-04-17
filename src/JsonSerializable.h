#ifndef JSONSERIALIZABLE_H
#define JSONSERIALIZABLE_H


#include <string>
#include <streambuf>

#include <Variant/Schema.h>
#include <Variant/Variant.h>

#include <boost/static_assert.hpp>

#include "LibVariantSupport.h"
#include "SchemaValidationException.h"


namespace Winzent {
    namespace ANN {


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
                std::extent<decltype(Winzent::ANN::JsonSchema<C>::schemaURI)>
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
                std::extent<decltype(Winzent::ANN::JsonSchema<C>::schemaURI)>
                    ::value > 1),
                int>::type = 0>
        inline C from_json(std::string& json)
        {
            libvariant::Variant jsonSchema(libvariant::DeserializeJSONFile(
                    Winzent::ANN::JsonSchema<C>::schemaURI));
            libvariant::Variant jsonData = libvariant::DeserializeJSON(
                    json);
            auto r(libvariant::SchemaValidate(
                    jsonSchema,
                    jsonData));

            if (r.Error()) {
                throw Winzent::ANN::SchemaValidationException(r);
                return;
            }

            return from_variant<C>();
        }


        template <
            class C,
            typename std::enable_if<(
                std::extent<decltype(Winzent::ANN::JsonSchema<C>::schemaURI)>
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
                std::extent<decltype(Winzent::ANN::JsonSchema<C>::schemaURI)>
                    ::value > 1),
                int>::type = 0>
        inline C* new_from_json(std::string& json)
        {
            auto jsonSchema(libvariant::DeserializeJSONFile(
                    Winzent::ANN::JsonSchema<C>::schemaURI));
            auto jsonData(libvariant::DeserializeJSON(json));
            auto r(libvariant::SchemaValidate(jsonSchema, jsonData));

            if (r.Error()) {
                throw Winzent::ANN::SchemaValidationException(r);
                return;
            }

            return new_from_variant<C>(jsonData);
        }


        template <class C>
        inline std::string to_json(C const& serializable)
        {
            return libvariant::SerializeJSON(to_variant<C>(serializable));
        }
    }
}

#endif // JSONSERIALIZABLE_H
