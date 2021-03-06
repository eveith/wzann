cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(wzann VERSION 0.6.1.3 LANGUAGES CXX)


include(CTest)
include(FindPkgConfig)
include(GNUInstallDirs)
include(CheckIncludeFileCXX)
find_library(readline_LIBRARY readline)
CHECK_INCLUDE_FILE_CXX("readline/readline.h" readline_HEADER)


find_package(GTest)
find_package(Boost 1.56.0 REQUIRED
    COMPONENTS program_options filesystem system)
find_package(LibVariant 1.0.0 REQUIRED)
pkg_check_modules(LIBWZALGORITHM libwzalgorithm>=0.8.0)
find_program(BATS bats)
find_program(A2X a2x)


set(WZANN_DATADIR
    "${CMAKE_INSTALL_FULL_DATADIR}/wzann"
    CACHE FILEPATH
    "General data director for wzann")
set(WZANN_SCHEMA_PATH
    "${WZANN_DATADIR}/schema"
    CACHE FILEPATH
    "Path to the JSON schema for serialized ANNs")


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VERSION")


add_custom_target(srctar
    COMMAND tar
        "-C" "${CMAKE_CURRENT_SOURCE_DIR}"
        "-vczf" "wzann-${wzann_VERSION}.tar.gz"
        "--xform='s,^,${PROJECT_NAME}-${wzann_VERSION}/,'"
        "--exclude-vcs" "--exclude-backups"
        "--exclude=Makefile" "--exclude='*.so*'"
        "lib" "bin" "man" "test" "CMakeLists.txt"
        "README.md" "VERSION.in" "COPYING")


add_subdirectory(lib)
add_subdirectory(bin)
enable_testing()
add_subdirectory(test)

if (${DOXYGEN_FOUND})
    add_subdirectory(doc)
endif()

if (EXISTS ${A2X})
    add_subdirectory(man)
else()
    message(WARNING "asciidoc not found, cannot build manpages.")
endif()
