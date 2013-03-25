#ifndef SIMULATEDANNEALINGTRAININGALGORITHMTEST_H
#define SIMULATEDANNEALINGTRAININGALGORITHMTEST_H

#include <QObject>

class SimulatedAnnealingTrainingAlgorithmTest : public QObject
{
    Q_OBJECT
public:
    explicit SimulatedAnnealingTrainingAlgorithmTest(QObject *parent = 0);
    
private slots:
    void testTrainXOR();

};

#endif // SIMULATEDANNEALINGTRAININGALGORITHMTEST_H
