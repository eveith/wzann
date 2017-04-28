set(test-wzann_SOURCES
    ClassRegistryTest.cpp

    NeuronTest.cpp
    LayerTest.cpp
    NeuralNetworkTest.cpp
    ActivationFunctionTest.cpp

    #BackpropagationTrainingAlgorithmTest.cpp
    #ElmanNetworkPatternTest.cpp
    #NeuralNetworkPatternTest.cpp
    #SimpleWeightRandomizerTest.cpp
    #NguyenWidrowWeightRandomizerTest.cpp
    #PerceptronNetworkPatternTest.cpp
    #PsoTrainingAlgorithmTest.cpp
    #REvolutionaryTrainingAlgorithmTest.cpp
    #RpropTrainingAlgorithmTest.cpp
    #SimulatedAnnealingTrainingAlgorithmTest.cpp
    #TrainingSetTest.cpp
    tst_ann.cpp)

set(test-wzann_HEADERS
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


set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_executable(tst_ann ${test-wzann_SOURCES})
add_test(tst_ann tst_ann)

target_include_directories(tst_ann
    PUBLIC ${wzann_SOURCE_DIR}/src
    PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(tst_ann
    stdc++
    gtest
    pthread
    wzann)


set_property(TARGET tst_ann PROPERTY CXX_STANDARD 14)
