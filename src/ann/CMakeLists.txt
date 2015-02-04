project(winzent-ann)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


set(winzent-ann_SOURCES
    ActivationFunction.cpp
    BackpropagationTrainingAlgorithm.cpp
    Connection.cpp
    ConstantActivationFunction.cpp
    ElmanNetworkPattern.cpp
    Exception.cpp
    Layer.cpp
    LinearActivationFunction.cpp
    NeuralNetwork.cpp
    NeuralNetworkPattern.cpp
    Neuron.cpp
    NguyenWidrowWeightRandomizer.cpp
    PerceptronNetworkPattern.cpp
    RememberingActivationFunction.cpp
    REvolutionaryTrainingAlgorithm.cpp
    RpropTrainingAlgorithm.cpp
    SigmoidActivationFunction.cpp
    SimulatedAnnealingTrainingAlgorithm.cpp
    TrainingAlgorithm.cpp
    TrainingSet.cpp
    Vector.cpp)

set(winzent-ann_HEADERS
    Winzent-ANN_global.h
    ActivationFunction.h
    BackpropagationTrainingAlgorithm.h
    Connection.h
    ConstantActivationFunction.h
    ElmanNetworkPattern.h
    Exception.h
    Layer.h
    LinearActivationFunction.h
    NeuralNetwork.h
    NeuralNetworkPattern.h
    Neuron.h
    NguyenWidrowWeightRandomizer.h
    PerceptronNetworkPattern.h
    RememberingActivationFunction.h
    REvolutionaryTrainingAlgorithm.h
    RpropTrainingAlgorithm.h
    SigmoidActivationFunction.h
    SimulatedAnnealingTrainingAlgorithm.h
    TrainingAlgorithm.h
    TrainingSet.h
    Vector.h)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core REQUIRED)
pkg_check_modules(log4cxx REQUIRED)


# Tweak for C++11:

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-std=c++1y)
endif()


add_library(winzent-ann
    SHARED ${winzent-ann_SOURCES})
add_dependencies(winzent-ann
    winzent-common)
target_link_libraries(winzent-ann
    Qt5::Core
    log4cxx
    winzent-common)
set_target_properties(winzent-ann
    PROPERTIES SOVERSION 0)
