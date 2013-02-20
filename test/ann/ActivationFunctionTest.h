#ifndef ACTIVATIONFUNCTIONTEST_H
#define ACTIVATIONFUNCTIONTEST_H


#include <QObject>


class ActivationFunctionTest: public QObject
{
    Q_OBJECT


public:


    explicit ActivationFunctionTest(QObject *parent = 0);
    

private slots:


    void testSigmoidActivationFunction();
};


#endif // ACTIVATIONFUNCTIONTEST_H
