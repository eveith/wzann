#ifndef CONNECTIONTEST_H
#define CONNECTIONTEST_H


#include <QObject>


class ConnectionTest: public QObject
{
    Q_OBJECT
public:
    explicit ConnectionTest(QObject *parent = 0);
    
private slots:
    void testConnectionMultiplicaton();
    void testFixedWeight();
};

#endif // CONNECTIONTEST_H
