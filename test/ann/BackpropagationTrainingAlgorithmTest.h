#ifndef BACKPROPAGATIONTRAININGALGORITHMTEST_H
#define BACKPROPAGATIONTRAININGALGORITHMTEST_H


#include <QObject>


class BackpropagationTrainingAlgorithmTest : public QObject
{
    Q_OBJECT
public:
    explicit BackpropagationTrainingAlgorithmTest(QObject *parent = 0);
    
private slots:
    void testTrainXOR();
};

#endif // BACKPROPAGATIONTRAININGALGORITHMTEST_H
