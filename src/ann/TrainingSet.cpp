/*
 * TrainingSet.cpp

 *
 *  Created on: 18.10.2012
 *      Author: eveith
 */


#include <cmath>

#include "TrainingSet.h"


namespace Winzent
{
    namespace ANN
    {


        TrainingItem::TrainingItem(
                ValueVector input,
                ValueVector expectedOutput):
                        m_input(ValueVector()),
                        m_expectedOutput(ValueVector())
        {
            // Copy all values so that we don't lose it and can modify them as
            // we wish:

            foreach (double i, input) {
                m_input.append(double(i));
            }

            foreach (double i, expectedOutput) {
                m_expectedOutput.append(double(i));
            }
        }


         TrainingItem::TrainingItem(const TrainingItem &copy):
                 m_input(ValueVector()),
                 m_expectedOutput(ValueVector())
         {
                foreach (double i, copy.m_input) {
                    m_input.append(double(i));
                }

                foreach (double i, copy.m_expectedOutput) {
                    m_expectedOutput.append(double(i));
                }
         }


        TrainingSet::TrainingSet(
                QList<TrainingItem> trainingData,
                double learningRate,
                double targetError,
                int maxNumEpochs):
                    m_trainingData(QList<TrainingItem>()),
                    m_learningRate(learningRate),
                    m_targetError(targetError),
                    m_maxNumEpochs(maxNumEpochs),
                    m_error(INFINITY)
        {
            foreach (TrainingItem i, trainingData) {
                m_trainingData << TrainingItem(i);
            }
        }


        TrainingSet::~TrainingSet()
        {
        }


        double TrainingSet::targetError() const
        {
            return m_targetError;
        }


        double TrainingSet::error() const
        {
            return m_error;
        }


        int TrainingSet::maxEpochs() const
        {
            return m_maxNumEpochs;
        }


        int TrainingSet::epochs() const
        {
            return m_epochs;
        }


        const QList<TrainingItem> TrainingSet::trainingData() const
        {
            return m_trainingData;
        }
    }
}
