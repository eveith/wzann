#ifndef TRAININGSETTEST_H
#define TRAININGSETTEST_H

#include <QObject>

class TrainingSetTest : public QObject
{
    Q_OBJECT
public:
    explicit TrainingSetTest(QObject *parent = 0);
    
private slots:
    void testOutputRelevant();
    void testJsonSerialization();
};

#endif // TRAININGSETTEST_H
