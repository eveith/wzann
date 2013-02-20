#ifndef ACTIVATIONFUNCTIONTEST_H
#define ACTIVATIONFUNCTIONTEST_H


#include "TestCase.h"


class ActivationFunctionTest: public TestCase
{
    Q_OBJECT


public:


    explicit ActivationFunctionTest(QObject *parent = 0);
    

private slots:


    void testSigmoidActivationFunction();
};


#endif // ACTIVATIONFUNCTIONTEST_H
