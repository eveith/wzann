#ifndef JSONSERIALIZABLE_H
#define JSONSERIALIZABLE_H

#include <QFile>
#include <QIODevice>

#include <QString>
#include <QJsonDocument>

#include <boost/static_assert.hpp>

#include <Variant/Variant.h>

#include "Clearable.h"
#include "ParserException.h"
#include "SchemaValidationException.h"

#include "common_global.h"


namespace Winzent {


    /*!
     * \brief The JsonSerializable interface defines serialization to and
     *  from JSON.
     *
     * Classes realizing this interface must implement a method to serialize
     * to JSON, and also one to load or re-load settings from the same
     * JSON representation. Those classes must consistently load the same
     * state they previously saved as long as it is the same source code
     * revision, i.e., `fromJSON(toJSON))` must yield exactly the same object.
     *
     * However, the implementation does not need to give any gurantees
     * about loading JSON-serialized data from earlier source code revisions.
     *
     * Generall, it is advisable that #clear() is called before #fromJSON()
     * is issued.
     */
    class COMMONSHARED_EXPORT JsonSerializable: public Clearable
    {
    public:


        /*!
         * \brief Serializes the object to JSON
         *
         * \return A QJsonDocument that represents the current state of the
         *  object.
         */
        virtual QJsonDocument toJSON() const = 0;


        /*!
         * \brief Restore a serialized state of the object from JSON.
         *
         * This re-initializes the object from a previously saved JSON
         * representation. It might call #clear() in order to reset the
         * object.
         *
         * \param[in] json The serialized JSON representation
         *
         * \throws ParserException If a syntactic error during parsing is
         *  encountered both syntactic or semantic
         *
         * \throws SchemaException If the supplied JSON document does not
         *  match the schema
         */
        virtual void fromJSON(const QJsonDocument &json) = 0;
    };


    /*!
     * \brief Schema URI strict for validation
     *
     * Classes that are serializable and have a JSON schema attached need
     * to provide a template specialization of this struct. Doing so enables
     * the automatic schema checking.
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
        typename std::enable_if<(std::is_base_of<Winzent::JsonSerializable, C>
                ::value
            && std::extent<decltype(Winzent::JsonSchema<C>::schemaURI)>
                 ::value <= 1),
            int>::type = 0>
    inline void deserialize(C& serializable, QJsonDocument const& json)
    {
        serializable.fromJSON(json);
    }


    template <
        class C,
        typename std::enable_if<(std::is_base_of<Winzent::JsonSerializable, C>
                ::value
            && std::extent<decltype(Winzent::JsonSchema<C>::schemaURI)>
                 ::value > 1),
            int>::type = 0>
    inline void deserialize(C& serializable, QJsonDocument const& json)
    {
        QFile jsonSchemaFile(Winzent::JsonSchema<C>::schemaURI);
        bool ok = jsonSchemaFile.open(QIODevice::ReadOnly|QIODevice::Text);

        if (! ok) {
            throw ParserException(QString(
                    "Could not open schema file at URI %1")
                    .arg(Winzent::JsonSchema<C>::schemaURI));
            return;
        }

        libvariant::Variant jsonSchema = libvariant::DeserializeJSON(
                jsonSchemaFile.readAll().data());
        libvariant::Variant jsonData = libvariant::DeserializeJSON(
                json.toJson().data());
        libvariant::SchemaResult r = libvariant::SchemaValidate(
                jsonSchema,
                jsonData);

        if (r.Error()) {
            throw Winzent::SchemaValidationException(r);
            return;
        }

        serializable.fromJSON(json);
    }


    template <
        class C,
        typename std::enable_if<(!std::is_base_of<Winzent::JsonSerializable,C>
                ::value),
            int>::type = 0>
    inline void deserialize(C &, QJsonDocument const &)
    {
    }


    template <class C>
    inline QJsonDocument serialize(C const& serializable)
    {
        return serializable.toJSON();
    }
}

#endif // JSONSERIALIZABLE_H
