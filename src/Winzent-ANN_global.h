#ifndef WINZENTANN_GLOBAL_H
#define WINZENTANN_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(WINZENTANN_LIBRARY)
#  define WINZENTANNSHARED_EXPORT Q_DECL_EXPORT
#else
#  define WINZENTANNSHARED_EXPORT Q_DECL_IMPORT
#endif


namespace Winzent {
    namespace ANN {
    }
}


#endif // WINZENTANN_GLOBAL_H
