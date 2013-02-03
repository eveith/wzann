/*!
 * \file	AbstractTrainingStrategy.cpp
 * \brief
 * \date	28.12.2012
 * \author	eveith
 */

#include "AbstractTrainingStrategy.h"


namespace Winzent
{
    namespace ANN
    {
        AbstractTrainingStrategy::AbstractTrainingStrategy(QObject *parent):
                QObject(parent)
        {
        }


        AbstractTrainingStrategy::~AbstractTrainingStrategy()
        {
        }
    }
}
