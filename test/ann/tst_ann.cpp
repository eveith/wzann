#include "Testrunner.h"


int main (int argc, char *argv[])
{
    return TestRunner::instance()->run(argc, argv);
}
