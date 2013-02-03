#ifndef ELMANNETWORKPATTERNTEST_H
#define ELMANNETWORKPATTERNTEST_H

#include <QObject>
#include <QList>

#include "ActivationFunction.h"


class ElmanNetworkPatternTest : public QObject
{
    Q_OBJECT

private:
    QList<int> layers;
    QList<Winzent::ANN::ActivationFunction*>
        activationFunctions;



public:
    explicit ElmanNetworkPatternTest(QObject *parent = 0);


private slots:
    void testConfigure();
};

#endif // ELMANNETWORKPATTERNTEST_H
