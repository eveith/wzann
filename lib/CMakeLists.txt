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

    ElmanNetworkPattern.cpp
    NeuralNetworkPattern.cpp
    PerceptronNetworkPattern.cpp

    SimpleWeightRandomizer.cpp
    NguyenWidrowWeightRandomizer.cpp

    TrainingSet.cpp
    TrainingItem.cpp
    TrainingAlgorithm.cpp
    GradientAnalysisHelper.cpp
    RpropTrainingAlgorithm.cpp
    BackpropagationTrainingAlgorithm.cpp)

set(wzann_wzalgorithm_SOURCES
   #PsoTrainingAlgorithm.cpp
   REvolutionaryTrainingAlgorithm.cpp)

set(wzann_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/WzannGlobal.h
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

    ElmanNetworkPattern.h
    NeuralNetworkPattern.h
    PerceptronNetworkPattern.h

    WeightRandomizer.h
    SimpleWeightRandomizer.h
    NguyenWidrowWeightRandomizer.h

    TrainingSet.h
    TrainingItem.h
    TrainingAlgorithm.h
    PsoTrainingAlgorithm.h
    GradientAnalysisHelper.h
    RpropTrainingAlgorithm.h
    REvolutionaryTrainingAlgorithm.h
    BackpropagationTrainingAlgorithm.h)

file(GLOB wzann_SCHEMATA schema/*.json)


if (${LIBWZALGORITHM_FOUND})
    list(APPEND wzann_SOURCES ${wzann_wzalgorithm_SOURCES})
endif()


set(CMAKE_INCLUDE_CURRENT_DIR ON)


if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
        OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
endif()


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/WzannGlobal.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/WzannGlobal.h")


SET(PKG_CONFIG_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
SET(PKG_CONFIG_INCLUDEDIR "\${prefix}/include")
SET(PKG_CONFIG_LIBS "-L\${libdir} ${wzann_LIBRARIES} -lwzann")
SET(PKG_CONFIG_CFLAGS "-I\${includedir}")

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/libwzann.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/libwzann.pc)


add_library(wzann SHARED ${wzann_SOURCES})

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_compile_definitions(wzann PUBLIC "WZANN_DEBUG")
endif()

target_include_directories(wzann
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${LIBVARIANT_INCLUDE_DIRS})

target_link_libraries(wzann
    PUBLIC ${LIBVARIANT_LIBRARIES}
    PUBLIC ${LIBWZALGORITHM_LIBRARIES})

set_target_properties(wzann
    PROPERTIES SOVERSION "${wzann_VERSION_MAJOR}.${wzann_VERSION_MINOR}"
    CXX_STANDARD 14)


if (${LIBWZALGORITHM_FOUND})
    target_include_directories(wzann PUBLIC ${libwzalgorithm_INCLUDE_DIRS})
    target_link_libraries(wzann PUBLIC ${libwzalgorithm_LIBRARIES})
endif()


install(TARGETS wzann
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${wzann_HEADERS} DESTINATION include/wzann)
install(FILES ${wzann_SCHEMATA} DESTINATION ${WZANN_SCHEMA_PATH})
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/libwzann.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
