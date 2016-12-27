cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(wzann VERSION 0.6.0 LANGUAGES CXX)


include(CTest)
include(FindPkgConfig)
include(GNUInstallDirs)


find_package(GTest)


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/wzann.spec.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wzann.spec")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VERSION")


add_custom_target(srctar
    COMMAND tar
        "-C" "${CMAKE_CURRENT_SOURCE_DIR}"
        "-vcJf" "wzann-${wzann_VERSION}.tar.xz"
        "--xform='s,^,${PROJECT_NAME}-${wzann_VERSION}/,'"
        "--exclude-vcs" "--exclude-backups"
        "--exclude=Makefile" "--exclude='*.so*'"
        "src" "test" "CMakeLists.txt"
        "README" "COPYING" "VERSION.in" "wzann.spec.in")


add_subdirectory(src)

if (${DOXYGEN_FOUND})
    add_subdirectory(doc)
endif()

if (${GTEST_FOUND})
    enable_testing()
    add_subdirectory(test)
else()
    message(AUTHOR_WARNING,
        "GTest was not found; no unit tests will be available and `make check' will not work.")
endif()
