#include <stdexcept>

#include "WeightFixedException.h"


namespace Winzent {
    namespace ANN {
        WeightFixedException::WeightFixedException():
                std::logic_error("Connection weight is fixed.")
        {
        }


        WeightFixedException::~WeightFixedException()
        {
        }
    } // namespace ANN
} // namespace Winzent
