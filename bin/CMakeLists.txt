include_directories(
    ${wzann_SOURCE_DIR}/lib
    ${wzann_BINARY_DIR}/lib
    ${Boost_INCLUDE_DIRS})


add_executable(wzann-mkann
    wzann-mkann.cpp)


set_target_properties(
    wzann-mkann
    PROPERTIES
        CXX_STANDARD 14)


target_link_libraries(wzann-mkann
    wzann
    ${Boost_LIBRARIES})
