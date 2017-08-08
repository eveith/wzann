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

if (${libwzalgorithm_FOUND})
    list(APPEND test-wzann_SOURCES
        #PsoTrainingAlgorithmTest.cpp
        REvolutionaryTrainingAlgorithmTest.cpp)
endif()


set(WZANN_TEST_SCHEMA_PATH "${wzann_SOURCE_DIR}/lib/schema")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/TestSchemaPath.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/TestSchemaPath.h")


    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    if (${GTEST_FOUND})

    add_executable(tst_ann ${test-wzann_SOURCES})
    add_test(tst_ann tst_ann)

    target_include_directories(tst_ann
        PUBLIC ${wzann_SOURCE_DIR}/lib
        PUBLIC ${wzann_BINARY_DIR}/lib
        PUBLIC ${Boost_INCLUDE_DIRS})

    target_link_libraries(tst_ann
        stdc++
        gtest
        pthread
        wzann)

    set_property(TARGET tst_ann PROPERTY CXX_STANDARD 14)
else()
    message(WARNING "GTest was not found; no unit tests will be available "
        "and `make check' will not work.")
endif()


if (EXISTS ${BATS})
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/test_FourBitParity.bats"
        "${CMAKE_CURRENT_BINARY_DIR}/test_FourBitParity.bats"
        @ONLY)
    add_test(
        NAME FourBitParity
        COMMAND bats "${CMAKE_CURRENT_BINARY_DIR}/test_FourBitParity.bats"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
else()
    message(WARNING "BATS was not found; will not test CLI tools.")
endif()
