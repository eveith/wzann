#-------------------------------------------------
#
# Project created by QtCreator 2013-02-01T17:02:48
#
#-------------------------------------------------

QT       -= gui

TARGET = winzentann
TEMPLATE = lib

DEFINES += WINZENTANN_LIBRARY
QMAKE_CXXFLAGS += -std=c++11

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += QJson

SOURCES += \
    TrainingSet.cpp \
    TrainingAlgorithm.cpp \
    SigmoidActivationFunction.cpp \
    RememberingActivationFunction.cpp \
    Neuron.cpp \
    NeuralNetworkPattern.cpp \
    NeuralNetwork.cpp \
    Exception.cpp \
    BackpropagationTrainingAlgorithm.cpp \
    ActivationFunction.cpp \
    ElmanNetworkPattern.cpp \
    ConstantActivationFunction.cpp \
    Layer.cpp \
    PerceptronNetworkPattern.cpp \
    Connection.cpp \
    LinearActivationFunction.cpp \
    EvolutionaryTrainingAlgorithm.cpp \
    SimulatedAnnealingTrainingAlgorithm.cpp \
    NguyenWidrowWeightRandomizer.cpp

HEADERS +=\
        Winzent-ANN_global.h \
    TrainingSet.h \
    TrainingAlgorithm.h \
    SigmoidActivationFunction.h \
    RememberingActivationFunction.h \
    Neuron.h \
    NeuralNetworkPattern.h \
    NeuralNetwork.h \
    Exception.h \
    BackpropagationTrainingAlgorithm.h \
    ActivationFunction.h \
    ElmanNetworkPattern.h \
    ConstantActivationFunction.h \
    Layer.h \
    PerceptronNetworkPattern.h \
    Connection.h \
    LinearActivationFunction.h \
    EvolutionaryTrainingAlgorithm.h \
    SimulatedAnnealingTrainingAlgorithm.h \
    NguyenWidrowWeightRandomizer.h

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}
