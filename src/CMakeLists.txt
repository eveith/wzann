set(wzann_SOURCES
    ClassRegistry.cpp
#    JsonSerializable.cpp
    ActivationFunction.cpp
#    BackpropagationTrainingAlgorithm.cpp
#    Connection.cpp
#    ConstantActivationFunction.cpp
#    ElmanNetworkPattern.cpp
#    Exception.cpp
    Layer.cpp
#    LinearActivationFunction.cpp
#    NeuralNetwork.cpp
#    NeuralNetworkPattern.cpp
    Neuron.cpp
#    SimpleWeightRandomizer.cpp
#    NguyenWidrowWeightRandomizer.cpp
#    PerceptronNetworkPattern.cpp
#    RememberingActivationFunction.cpp
#    PsoTrainingAlgorithm.cpp
#    REvolutionaryTrainingAlgorithm.cpp
#    RpropTrainingAlgorithm.cpp
#    SigmoidActivationFunction.cpp
#    SymmetricSigmoidActivationFunction.cpp
#    SimulatedAnnealingTrainingAlgorithm.cpp
#    TanhActivationFunction.cpp
#    TrainingAlgorithm.cpp
#    TrainingSet.cpp
#    Vector.cpp
)

set(wzann_HEADERS
    WzannGlobal.h
    enum.h
    ClassRegistry.h
    JsonSerializable.h
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

set(wzann_RESOURCE_FILES
    Schemata.qrc)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
endif()


add_library(wzann ${wzann_SOURCES})

target_include_directories(wzann
    #PUBLIC ${winzent-common_SOURCE_DIR}
    #PUBLIC ${winzent-algorithm_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${LIBVARIANT_INCLUDE_DIRS})

target_link_libraries(wzann
    #winzent-common
    #winzent-algorithm
    PUBLIC ${LIBVARIANT_LIBRARIES})

set_target_properties(wzann
    PROPERTIES SOVERSION "${wzann_VERSION_MAJOR}.${wzann_VERSION_MINOR}"
    CXX_STANDARD 14)

install(TARGETS wzann
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${wzann_HEADERS} DESTINATION include/winzent/ann)
