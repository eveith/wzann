#ifndef SIMPLEWEIGHTRANDOMIZERTEST_H
#define SIMPLEWEIGHTRANDOMIZERTEST_H


#include <QObject>


class SimpleWeightRandomizerTest: public QObject
{
    Q_OBJECT

public:
    SimpleWeightRandomizerTest(QObject *parent = nullptr);

private slots:
    void testWeightRandomization();
};


#endif // SIMPLEWEIGHTRANDOMIZERTEST_H
