#ifndef NGUYENWIDROWWEIGHTRANDOMIZERTEST_H
#define NGUYENWIDROWWEIGHTRANDOMIZERTEST_H

#include <QObject>

class NguyenWidrowWeightRandomizerTest: public QObject
{
    Q_OBJECT
public:
    explicit NguyenWidrowWeightRandomizerTest(QObject *parent = 0);
    
private slots:
    void testRandomizeWeights();
};

#endif // NGUYENWIDROWWEIGHTRANDOMIZERTEST_H
