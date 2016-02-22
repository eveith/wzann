/*!
 * \file	BasicException.cpp
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */


#include <string>
#include <QString>

#include "Exception.h"


namespace Winzent {
    namespace ANN {


        LayerSizeMismatchException::LayerSizeMismatchException(
                int actualSize,
                int expectedSize):
                    BasicException(),
                    actualSize(actualSize),
                    expectedSize(expectedSize)
        {
            m_what = QString("Layer sizes mismatch: "
                        "Expected %1 item(s), got %2.")
                    .arg(expectedSize).arg(actualSize).toStdString();
        }


        const char *LayerSizeMismatchException::what() const noexcept
        {
            return m_what.c_str();
        }
    }
}
