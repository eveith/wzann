#ifndef SCHEMAVALIDATIONEXCEPTION_H
#define SCHEMAVALIDATIONEXCEPTION_H


#include <stdexcept>

#include <Variant/Schema.h>


namespace wzann {


    /*!
     * \brief The SchemaValidationException class indicates an error
     *  during serialization/de-serialization to JSON when the JSON data
     *  does not conform to the supplied JSON schema.
     */
    class SchemaValidationException: public std::runtime_error
    {

    public:


        SchemaValidationException(libvariant::SchemaResult r);

        virtual ~SchemaValidationException();


        libvariant::SchemaResult const& schemaResult() const;

    private:


        libvariant::SchemaResult m_schemaResult;
    };
} // namespace wzann

#endif // SCHEMAVALIDATIONEXCEPTION_H
