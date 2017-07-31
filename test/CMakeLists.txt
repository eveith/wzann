set(test-wzann_SOURCES
    ClassRegistryTest.cpp

    NeuronTest.cpp
    LayerTest.cpp
    NeuralNetworkTest.cpp
    ActivationFunctionTest.cpp

    NeuralNetworkPatternTest.cpp
    ElmanNetworkPatternTest.cpp
    PerceptronNetworkPatternTest.cpp

    SimpleWeightRandomizerTest.cpp
    NguyenWidrowWeightRandomizerTest.cpp

    #PsoTrainingAlgorithmTest.cpp
    #REvolutionaryTrainingAlgorithmTest.cpp
    TrainingSetTest.cpp
    RpropTrainingAlgorithmTest.cpp
    BackpropagationTrainingAlgorithmTest.cpp
    #SimulatedAnnealingTrainingAlgorithmTest.cpp
    tst_ann.cpp)

set(test-wzann_HEADERS
    TestSchemaPath.h
    ClassRegistryTest.h
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
    TrainingSetTest.h)


get_filename_component(
    WZANN_TEST_SCHEMA_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/schema"
    REALPATH)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/TestSchemaPath.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TestSchemaPath.h")


set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_executable(tst_ann ${test-wzann_SOURCES})
add_test(tst_ann tst_ann)

target_include_directories(tst_ann
    PUBLIC ${wzann_SOURCE_DIR}/src
    PUBLIC ${wzann_BINARY_DIR}/src
    PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(tst_ann
    stdc++
    gtest
    pthread
    wzann)


set_property(TARGET tst_ann PROPERTY CXX_STANDARD 14)
