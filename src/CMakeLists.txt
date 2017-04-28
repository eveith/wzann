set(wzann_SOURCES
    ClassRegistry.cpp
    LibVariantSupport.cpp
    JsonSerializable.cpp

    WeightFixedException.cpp
    NoConnectionException.cpp
    UnknownNeuronException.cpp
    SchemaValidationException.cpp
    LayerSizeMismatchException.cpp

    Layer.cpp
    Neuron.cpp
    Vector.cpp
    Connection.cpp
    NeuralNetwork.cpp
    ActivationFunction.cpp

    NeuralNetworkPattern.cpp

#    BackpropagationTrainingAlgorithm.cpp
#    ElmanNetworkPattern.cpp
#    Exception.cpp
#    LinearActivationFunction.cpp
#    SimpleWeightRandomizer.cpp
#    NguyenWidrowWeightRandomizer.cpp
#    PerceptronNetworkPattern.cpp
#    PsoTrainingAlgorithm.cpp
#    REvolutionaryTrainingAlgorithm.cpp
#    RpropTrainingAlgorithm.cpp
#    SimulatedAnnealingTrainingAlgorithm.cpp
#    TrainingAlgorithm.cpp
#    TrainingSet.cpp
)

set(wzann_HEADERS
    WzannGlobal.h
    enum.h

    ClassRegistry.h
    JsonSerializable.h
    LibVariantSupport.h

    WeightFixedException.h
    NoConnectionException.h
    UnknownNeuronException.h
    SchemaValidationException.h
    LayerSizeMismatchException.h

    Layer.h
    Neuron.h
    Vector.h
    Connection.h
    NeuralNetwork.h
    ActivationFunction.h

    NeuralNetworkPattern.h

    BackpropagationTrainingAlgorithm.h
    ElmanNetworkPattern.h
    LinearActivationFunction.h
    WeightRandomizer.h
    SimpleWeightRandomizer.h
    NguyenWidrowWeightRandomizer.h
    PerceptronNetworkPattern.h
    QtContainerOutput.h
    PsoTrainingAlgorithm.h
    REvolutionaryTrainingAlgorithm.h
    RpropTrainingAlgorithm.h
    TrainingAlgorithm.h
    TrainingSet.h)

set(wzann_RESOURCE_FILES
    Schemata.qrc)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
endif()


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/WzannGlobal.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/WzannGlobal.h")


add_library(wzann ${wzann_SOURCES})

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_compile_definitions(wzann PUBLIC "WZANN_DEBUG")
endif()

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
