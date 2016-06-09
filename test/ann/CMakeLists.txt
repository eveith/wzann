project(test-winzent-ann CXX)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


set(test-winzent-ann_SOURCES
    ActivationFunctionTest.cpp
    BackpropagationTrainingAlgorithmTest.cpp
    ElmanNetworkPatternTest.cpp
    LayerTest.cpp
    NeuralNetworkPatternTest.cpp
    NeuralNetworkTest.cpp
    NeuronTest.cpp
    SimpleWeightRandomizerTest.cpp
    NguyenWidrowWeightRandomizerTest.cpp
    PerceptronNetworkPatternTest.cpp
    PsoTrainingAlgorithmTest.cpp
    REvolutionaryTrainingAlgorithmTest.cpp
    RpropTrainingAlgorithmTest.cpp
    SimulatedAnnealingTrainingAlgorithmTest.cpp
    Testrunner.cpp
    TrainingSetTest.cpp
    tst_ann.cpp)

set(test-winzent-ann_HEADERS
    ActivationFunctionTest.h
    BackpropagationTrainingAlgorithmTest.h
    ElmanNetworkPatternTest.h
    LayerTest.h
    NeuralNetworkPatternTest.h
    NeuralNetworkTest.h
    NeuronTest.h
    SimpleWeightRandomizerTest.h
    NguyenWidrowWeightRandomizerTest.h
    PerceptronNetworkPatternTest.h
    PsoTrainingAlgorithmTest.h
    REvolutionaryTrainingAlgorithmTest.h
    RpropTrainingAlgorithmTest.h
    SimulatedAnnealingTrainingAlgorithmTest.h
    Testrunner.h
    TrainingSetTest.h)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Test REQUIRED)
find_package(Qt5Sql REQUIRED)
find_package(Qt5Concurrent REQUIRED)


# Tweak for C++11:

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_compile_options(-std=c++1y)
endif()


add_executable(tst_ann
    ${test-winzent-ann_SOURCES})
add_test(tst_ann
    tst_ann)
target_include_directories(
    tst_ann
    PRIVATE ${winzent-ann_SOURCE_DIR}
    PRIVATE ${winzent-ann_INCLUDE_DIRS})
add_dependencies(tst_ann
    winzent-common
    winzent-ann)
target_link_libraries(tst_ann
    stdc++
    Qt5::Core
    Qt5::Test
    Qt5::Sql
    Qt5::Concurrent
    ${log4cxx_LIBRARIES}
    winzent-ann)

