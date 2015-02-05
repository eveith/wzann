project(test-winzent-ann CXX)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


set(test-winzent-ann_SOURCES
    ActivationFunctionTest.cpp
    BackpropagationTrainingAlgorithmTest.cpp
    ConnectionTest.cpp
    ElmanNetworkPatternTest.cpp
    LayerTest.cpp
    NeuralNetworkPatternTest.cpp
    NeuralNetworkTest.cpp
    NeuronTest.cpp
    NguyenWidrowWeightRandomizerTest.cpp
    PerceptronNetworkPatternTest.cpp
    REvolutionaryTrainingAlgorithmTest.cpp
    RpropTrainingAlgorithmTest.cpp
    SimulatedAnnealingTrainingAlgorithmTest.cpp
    Testrunner.cpp
    TrainingSetTest.cpp
    tst_ann.cpp)

set(test-winzent-ann_HEADERS
    ActivationFunctionTest.h
    BackpropagationTrainingAlgorithmTest.h
    ConnectionTest.h
    ElmanNetworkPatternTest.h
    LayerTest.h
    NeuralNetworkPatternTest.h
    NeuralNetworkTest.h
    NeuronTest.h
    NguyenWidrowWeightRandomizerTest.h
    PerceptronNetworkPatternTest.h
    REvolutionaryTrainingAlgorithmTest.h
    RpropTrainingAlgorithmTest.h
    SimulatedAnnealingTrainingAlgorithmTest.h
    Testrunner.h
    TrainingSetTest.h)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Test REQUIRED)


# Tweak for C++11:

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-std=c++1y)
endif()


include_directories(../../src/ann)


add_executable(tst_ann
    ${test-winzent-ann_SOURCES})
add_test(tst_ann
    tst_ann)
add_dependencies(tst_ann
    winzent-common
    winzent-ann)
target_link_libraries(tst_ann
    stdc++
    Qt5::Core
    Qt5::Test
    log4cxx
    winzent-ann)

