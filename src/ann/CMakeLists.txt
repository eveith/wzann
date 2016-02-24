project(winzent-ann)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(winzent-ann_VERSION_MAJOR ${WINZENT_VERSION_MAJOR})
set(winzent-ann_VERSION_MINOR 2)
set(winzent-ann_VERSION_PATCHLEVEL 0)
set(winzent-ann_VERSION
    "${winzent-ann_VERSION_MAJOR}.${winzent_ann_VERSION_MINOR}.${winzent_ann_VERSION_PATCHLEVEL}")
set(winzent-ann_SOVERSION
    "${winzent-ann_VERSION_MAJOR}.${winzent-ann_VERSION_MINOR}")


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
    SimpleWeightRandomizer.cpp
    NguyenWidrowWeightRandomizer.cpp
    PerceptronNetworkPattern.cpp
    RememberingActivationFunction.cpp
    PsoTrainingAlgorithm.cpp
    REvolutionaryTrainingAlgorithm.cpp
    RpropTrainingAlgorithm.cpp
    SigmoidActivationFunction.cpp
    SymmetricSigmoidActivationFunction.cpp
    SimulatedAnnealingTrainingAlgorithm.cpp
    TanhActivationFunction.cpp
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
    WeightRandomizer.h
    SimpleWeightRandomizer.h
    NguyenWidrowWeightRandomizer.h
    PerceptronNetworkPattern.h
    QtContainerOutput.h
    RememberingActivationFunction.h
    PsoTrainingAlgorithm.h
    REvolutionaryTrainingAlgorithm.h
    RpropTrainingAlgorithm.h
    SigmoidActivationFunction.h
    SymmetricSigmoidActivationFunction.h
    SimulatedAnnealingTrainingAlgorithm.h
    TanhActivationFunction.h
    TrainingAlgorithm.h
    TrainingSet.h
    Vector.h)

set(winzent-ann_RESOURCE_FILES
    Schemata.qrc)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Sql REQUIRED)
find_package(Qt5Concurrent REQUIRED)

find_package(Boost 1.56.0 REQUIRED)
pkg_check_modules(LOG4CXX REQUIRED liblog4cxx>=0.10.0)

include_directories(../common)


# Tweak for C++14:

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_compile_options(-std=c++1y)
endif()


qt5_add_resources(winzent-ann_RESOURCES
    ${winzent-ann_RESOURCE_FILES})
add_library(winzent-ann
    SHARED ${winzent-ann_SOURCES} ${winzent-ann_RESOURCES})
target_include_directories(winzent-ann
    PUBLIC ${winzent-common_SOURCE_DIR}
    PUBLIC ${winzent-algorithm_SOURCE_DIR}
    PUBLIC ${BOOST_INCLUDE_DIRS}
    PUBLIC ${LOG4CXX_INCLUDE_DIRS})
target_link_libraries(winzent-ann
    Qt5::Core
    Qt5::Sql
    Qt5::Concurrent
    ${LOG4CXX_LIBRARIES}
    winzent-common
    winzent-algorithm)
set_target_properties(winzent-ann
    PROPERTIES SOVERSION ${winzent-ann_SOVERSION})

install(TARGETS winzent-ann LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${winzent-ann_HEADERS} DESTINATION include/winzent/ann)
