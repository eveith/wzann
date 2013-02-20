#ifndef PERCEPTRONNETWORKPATTERNTEST_H
#define PERCEPTRONNETWORKPATTERNTEST_H


#include <QObject>


class PerceptronNetworkPatternTest : public QObject
{
    Q_OBJECT
public:
    explicit PerceptronNetworkPatternTest(QObject *parent = 0);
    
signals:
    
private slots:

    void testConfigure();

    /*!
     * Tests that the calcuation function works by creating the standard
     * XOR perceptron (in a state where it has already learnt) and running
     * inputs through it.
     */
    void testCalculate();
    
};

#endif // PERCEPTRONNETWORKPATTERNTEST_H
