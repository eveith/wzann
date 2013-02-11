#include <cstdlib>
#include <iostream>

#include <QObject>
#include <QtTest>

#include "NeuronTest.h"
#include "NeuralNetworkTest.h"
#include "NeuralNetworkPatternTest.h"
#include "ElmanNetworkPatternTest.h"


using std::cout;
using std::endl;


void printHelp()
{
    cout << "Usage: tst_ann [ -l | -h | test name ]" << endl
            << endl
            << "test name   Name of a particular test class" << endl
            << "-l          List all tests available" << endl
            << "-h          Print this help" << endl;
}


int main (int argc, char *argv[])
{
    // Build test classes list:

    QList<QObject*> testClasses;
    testClasses
            << new NeuronTest()
            << new NeuralNetworkTest()
            << new NeuralNetworkPatternTest()
            << new ElmanNetworkPatternTest();

    // Check command line parameters and run tests:

    if (2 <= argc) {
        if(0 == strcmp(argv[1], "-h")) {
            printHelp();
            return EXIT_SUCCESS;
        } else if (0 == strcmp(argv[1], "-l")) {
            // Print all class names:

            foreach (QObject *o, testClasses) {
                cout << o->metaObject()->className() << endl;
            }

            return EXIT_SUCCESS;
        } else {
            foreach (QObject *o, testClasses) {
                if (0 == strcmp(o->metaObject()->className(), argv[1])) {
                    QStringList args;

                    for (int i = 2; i < argc; ++i) {
                        args << argv[i];
                    }

                    return QTest::qExec(o, args);
                }
            }
        }
    } else if (1 == argc) {
        // Run all tests:

        QStringList args;
        for (int i = 0; i < argc; ++i) {
            args << argv[i];
        }

        int rc = 0;

        foreach (QObject *o, testClasses) {
            rc |= QTest::qExec(o, args);
        }

        return rc;
    } else {
        printHelp();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
