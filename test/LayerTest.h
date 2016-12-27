#ifndef LAYERTEST_H
#define LAYERTEST_H

#include <QObject>


class LayerTest : public QObject
{
    Q_OBJECT
public:
    explicit LayerTest(QObject *parent = 0);
    
signals:
    
private slots:
    void testLayerCreation();
    void testNeuronAddition();
    void testNeuronIterator();
};

#endif // LAYERTEST_H
