#include <cstdio>
#include "Testrunner.h"


TestRunner *TestRunner::m_instance = NULL;


TestRunner::TestRunner(): m_testcases(QList<QObject*>())
{
}


TestRunner* TestRunner::instance()
{
    if (NULL == TestRunner::m_instance) {
        m_instance = new TestRunner();
    }

    return m_instance;
}



bool TestRunner::addTestcase(QObject *testcase)
{
    if (!m_testcases.contains(testcase)) {
        m_testcases << testcase;
    }
    return true;
}


void TestRunner::printTestcasesList() const
{
    foreach (QObject *o, m_testcases) {
        printf("%s\n", o->metaObject()->className());
    }
}


void TestRunner::printHelp(const char *basename) const
{
    printf( "Usage: %s [ -l | -h | test name [ test args... ]]\n"
            "\n"
            "-l          List all tests available\n"
            "-h          Prints this help\n"
            "test name   Name of a particular test class\n",
            basename);
}


int TestRunner::run(int argc, char *argv[])
{
    int rc = 0;

    if (1 == argc) {
        // No argument given, run all testcases:

        foreach (QObject *o, m_testcases) {
            rc |= QTest::qExec(o);
        }
    } else {

        // Look for the command line parameters we know:

        if(0 == strcmp("-h", argv[1])) {
            printHelp(argv[0]);
            return 0;
        } else if (0 == strcmp("-l", argv[1])) {
            printTestcasesList();
            return 0;
        }

        // Otherwise, assume it's the name of a testcase and try to find it.

        foreach (QObject *o, m_testcases) {
            if (0 == strcmp(argv[1], o->metaObject()->className())) {
                QStringList args;

                for (int i = 1; i != argc; ++i) {
                   args << argv[i];
                }

                rc = QTest::qExec(o, args);
            }
        }
    }


    return rc;
}

