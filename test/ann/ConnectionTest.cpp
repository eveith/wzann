#include "Connection.h"
#include "Exception.h"

#include "Testrunner.h"
#include "ConnectionTest.h"


using Winzent::ANN::Connection;
using Winzent::ANN::WeightFixedException;


ConnectionTest::ConnectionTest(QObject *parent) :
    QObject(parent)
{
}


void ConnectionTest::testConnectionMultiplicaton()
{
    Connection c(nullptr, nullptr, 0.5);
    QCOMPARE(c * 1.0, 0.5);
}


void ConnectionTest::testFixedWeight()
{
    Connection c(NULL, NULL, 0.5);

    c.weight(1.0);
    QCOMPARE(c.weight(), 1.0);

    c.fixedWeight(true);
    try {
        c.weight(2.0);
        QFAIL("Fixed weights can still be modified");
    } catch (WeightFixedException) {
        QCOMPARE(c.weight(), 1.0);
    }
}


TESTCASE(ConnectionTest)
