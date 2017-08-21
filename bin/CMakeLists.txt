include_directories(
    ${wzann_SOURCE_DIR}/lib
    ${wzann_BINARY_DIR}/lib
    ${Boost_INCLUDE_DIRS})


add_executable(wzann-mkann
    wzann-mkann.cpp)

add_executable(wzann-train
    wzann-train.cpp)


set_target_properties(
    wzann-mkann
    wzann-train
    PROPERTIES
        CXX_STANDARD 14)


target_link_libraries(wzann-mkann
    wzann
    ${Boost_LIBRARIES})

target_link_libraries(wzann-train
    wzann
    ${Boost_LIBRARIES})


install(TARGETS wzann-mkann wzann-train DESTINATION ${CMAKE_INSTALL_BINDIR})


find_library(readline_LIBRARY readline)
CHECK_INCLUDE_FILE_CXX("readline/readline.h" readline_HEADER)

if (EXISTS ${readline_LIBRARY} AND ${readline_HEADER})
    add_executable(wzann-repl
        wzann-repl.cpp)
    set_target_properties(wzann-repl PROPERTIES CXX_STANDARD 14)

    target_link_libraries(wzann-repl
        wzann
        readline
        ${Boost_LIBRARIES})

    install(TARGETS wzann-repl DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    message("Did not find readline; will not build wzann-repl.")
endif()
