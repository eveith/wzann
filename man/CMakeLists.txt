set(wzann_MANPAGE_SOURCES
    wzann-mkann.1.txt
    wzann-train.1.txt)
set(a2x_common_options
    -d manpage -f manpage --asciidoc-opts='-o ${CMAKE_CURRENT_BINARY_DIR}')


if (EXISTS ${readline_LIBRARY} AND ${readline_HEADER})
    LIST(APPEND wzann_MANPAGE_SOURCES wzann-repl.1.txt)
endif()


foreach (man_src ${wzann_MANPAGE_SOURCES})
    string(REGEX REPLACE "\\.[^.]*$" "" man_dst ${man_src})
    list(APPEND wzann_MANPAGES ${man_dst})
    add_custom_command(
        OUTPUT ${man_dst}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Creating ${man_dst}..."
        COMMAND ${A2X} ${a2x_common_options}
            "${CMAKE_CURRENT_SOURCE_DIR}/${man_src}")
endforeach()

add_custom_target(man DEPENDS ${wzann_MANPAGES})
