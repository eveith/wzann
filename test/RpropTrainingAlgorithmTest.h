#ifndef RPROPTRAININGALGORITHMTEST_H
#define RPROPTRAININGALGORITHMTEST_H


#include <QObject>


class RpropTrainingAlgorithmTest: public QObject
{
    Q_OBJECT

public:
    explicit RpropTrainingAlgorithmTest(QObject *parent = 0);

private slots:
    void testTrainXOR();
};

#endif // RPROPTRAININGALGORITHMTEST_H
