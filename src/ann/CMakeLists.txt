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
find_package(Qt5Sql REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Boost 1.56.0 REQUIRED)
pkg_check_modules(LOG4CXX REQUIRED liblog4cxx>=0.10.0)


# Tweak for C++14:

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-std=c++1y)
endif()


add_library(winzent-ann
    SHARED ${winzent-ann_SOURCES})
add_dependencies(winzent-ann
    winzent-common)
target_link_libraries(winzent-ann
    Qt5::Core
    Qt5::Sql
    Qt5::Concurrent
    ${LOG4CXX_LIBRARIES}
    winzent-common)
set_target_properties(winzent-ann
    PROPERTIES SOVERSION ${winzent-ann_SOVERSION})

install(TARGETS winzent-ann LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${winzent-ann_HEADERS} DESTINATION include/winzent/ann)
