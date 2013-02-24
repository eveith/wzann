#ifndef TESTRUNNER_H
#define TESTRUNNER_H

#include <QObject>
#include <QtTest>


class TestRunner
{

private:


    static TestRunner *m_instance;

    QList<QObject*> m_testcases;


    TestRunner();


    /*!
     * Prints an help screen to STDOUT.
     */
    void printHelp(const char *basename) const;


    /*!
     * Prints a list of all testcases available.
     */
    void printTestcasesList() const;


public:


    static TestRunner* instance();


    bool addTestcase(QObject *testcase);


    /*!
     * Runs all or only one test, given the command-line parameters.
     */
    int run(int argc, char *argv[]);
};



/*!
 * This macro auto-adds a class as a testcase to the test suite.
 */
#define TESTCASE(klass)\
    static bool __##klass##_registered =\
    TestRunner::instance()->addTestcase(new klass());

#endif // TESTRUNNER_H
