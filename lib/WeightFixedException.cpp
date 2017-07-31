#include <stdexcept>

#include "WeightFixedException.h"


namespace wzann {
    WeightFixedException::WeightFixedException():
            std::logic_error("Connection weight is fixed.")
    {
    }


    WeightFixedException::~WeightFixedException()
    {
    }
} // namespace wzann
