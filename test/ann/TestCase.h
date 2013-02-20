#ifndef TESTCASE_H
#define TESTCASE_H


#include <QObject>
#include "tst_ann.h"


class TestCase: public QObject
{
    Q_OBJECT

private:

    static bool _registered;


public:


    TestCase(QObject *parent = 0): QObject(parent){}
};


#define TESTCASE(klass)\
    static bool __##klass##_registered =\
    TestRunner::instance()->addTestcase(new klass());


#endif // TESTCASE_H
