# hilum — Shared C API for LLM inference
#
# Produces a static library (libhilum) that wraps llama.cpp, mtmd, and
# json-schema-to-grammar behind a stable C interface.
#
# Dependencies: llama, ggml, mtmd (all built from the parent engine repo).
# Does NOT require the full `common` library.

set(TARGET hilum)

add_library(${TARGET} STATIC
    hilum_llm.cpp
    hilum_llm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/json-schema-to-grammar.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE HILUM_BUILD)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

find_package(Threads REQUIRED)

# Ensure mtmd target exists (consumers may have already added it)
if (NOT TARGET mtmd)
    if(NOT DEFINED LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION "0.0.0")
    endif()
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
        ${CMAKE_CURRENT_BINARY_DIR}/mtmd
        EXCLUDE_FROM_ALL
    )
endif()

target_link_libraries(${TARGET} PUBLIC llama)
target_link_libraries(${TARGET} PRIVATE mtmd Threads::Threads)
