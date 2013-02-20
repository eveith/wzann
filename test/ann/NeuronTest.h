/*!
 * \file	NeuronTest.h
 * \brief
 * \date	11.01.2013
 * \author	eveith
 */


#ifndef NEURONTEST_H_
#define NEURONTEST_H_


#include <qobject.h>


class NeuronTest: public QObject
{
    Q_OBJECT


private slots:

    void testClone();
};

#endif /* NEURONTEST_H_ */
