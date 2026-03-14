# hilum — Shared C API for LLM inference
#
# Produces a static library (libhilum) that wraps llama.cpp, mtmd, and
# json-schema-to-grammar behind a stable C interface.
#
# Dependencies: llama, ggml, mtmd (all built from the parent engine repo).
# Does NOT require the full `common` library.

set(TARGET hilum)

set(HILUM_MTMD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tools/mtmd")
if (NOT EXISTS "${HILUM_MTMD_SOURCE_DIR}/CMakeLists.txt")
    set(HILUM_MTMD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mtmd")
endif()

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
    ${HILUM_MTMD_SOURCE_DIR}
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE HILUM_BUILD)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

find_package(Threads REQUIRED)

# Ensure mtmd target exists for standalone libhilum consumers.
# When the normal tools build is enabled, top-level CMake adds tools/ later and
# defines mtmd there, so avoid creating a duplicate target here.
if (NOT TARGET mtmd AND NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS))
    if(NOT DEFINED LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION "0.0.0")
    endif()
    add_subdirectory(
        ${HILUM_MTMD_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/mtmd
        EXCLUDE_FROM_ALL
    )
endif()

target_link_libraries(${TARGET} PUBLIC llama)
target_link_libraries(${TARGET} PRIVATE mtmd Threads::Threads)
