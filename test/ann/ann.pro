#-------------------------------------------------
#
# Project created by QtCreator 2012-09-17T18:27:40
#
#-------------------------------------------------


QT      += testlib

QT      -= gui

TARGET  = tst_ann
CONFIG  += console
CONFIG  -= app_bundle
CONFIG  += testcase

TEMPLATE = app


INCLUDEPATH += \
    ../../src/ann

TEST_LIB_PATHS += ../../src/ann

LIBS += -L../../src/ann -lwinzentann
QMAKE_CXXFLAGS += -std=c++11


HEADERS += \
    NeuronTest.h \
    NeuralNetworkTest.h \
    NeuralNetworkPatternTest.h \
    ElmanNetworkPatternTest.h \
    LayerTest.h \
    PerceptronNetworkPatternTest.h \
    ActivationFunctionTest.h \
    Testrunner.h \
    ConnectionTest.h \
    mock/LinearNeuralNetworkPattern.h \
    BackpropagationTrainingAlgorithmTest.h \
    SimulatedAnnealingTrainingAlgorithmTest.h \
    NguyenWidrowWeightRandomizerTest.h \
    TrainingSetTest.h \
    REvolutionaryTrainingAlgorithmTest.h
    
SOURCES += \
    tst_ann.cpp \
    NeuronTest.cpp \
    NeuralNetworkTest.cpp \
    NeuralNetworkPatternTest.cpp \
    ElmanNetworkPatternTest.cpp \
    LayerTest.cpp \
    PerceptronNetworkPatternTest.cpp \
    ActivationFunctionTest.cpp \
    Testrunner.cpp \
    ConnectionTest.cpp \
    mock/LinearNeuralNetworkPattern.cpp \
    BackpropagationTrainingAlgorithmTest.cpp \
    SimulatedAnnealingTrainingAlgorithmTest.cpp \
    NguyenWidrowWeightRandomizerTest.cpp \
    TrainingSetTest.cpp \
    REvolutionaryTrainingAlgorithmTest.cpp

DEFINES += SRCDIR=\\\"$$PWD/\\\"

QMAKE_CLEAN += *.out *.png *.svg *.pdf */*.out
QMAKE_CLEAN += -r *Test/


include( ../../3rdparty/QMakeTestRunner/testtarget.pri )
