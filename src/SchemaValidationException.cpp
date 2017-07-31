#include <stdexcept>

#include <Variant/Schema.h>

#include "SchemaValidationException.h"


namespace wzann {
    SchemaValidationException::SchemaValidationException(
            libvariant::SchemaResult r):
                std::runtime_error(r.PrettyPrintMessage().c_str()),
                m_schemaResult(r)
    {
    }


    SchemaValidationException::~SchemaValidationException()
    {
    }


    libvariant::SchemaResult const&
    SchemaValidationException::schemaResult() const
    {
        return m_schemaResult;
    }
} // namespace wzann
