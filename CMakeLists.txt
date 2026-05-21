cmake_minimum_required(VERSION 3.20)
project(c2d C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Wunused-macros
    -fno-strict-aliasing
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -fno-math-errno -ffp-contract=fast)
endif()

# OpenMP for per-tile parallelism in the encoder (optional; falls back to
# serial when not available).
find_package(OpenMP)

add_library(c2d STATIC c2d.c)
target_include_directories(c2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(c2d PUBLIC m)
if(OpenMP_C_FOUND)
    target_link_libraries(c2d PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(c2d PUBLIC C2D_HAVE_OPENMP=1)
endif()

add_executable(c2d_test c2d_test.c)
target_link_libraries(c2d_test PRIVATE c2d)

add_executable(c2d_perf c2d_perf.c)
target_link_libraries(c2d_perf PRIVATE c2d)

add_executable(c2d_bench c2d_bench.c)
target_link_libraries(c2d_bench PRIVATE c2d)
# stb_image triggers many narrowing/sign warnings; silence for this TU only.
target_compile_options(c2d_bench PRIVATE -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-parameter -Wno-implicit-fallthrough -Wno-cast-qual -Wno-sign-compare -Wno-misleading-indentation)

# c2d CLI: encode/decode/info commands. Output name is "c2d" (matches user expectation).
add_executable(c2d_cli c2d_cli.c)
set_target_properties(c2d_cli PROPERTIES OUTPUT_NAME "c2d")
target_link_libraries(c2d_cli PRIVATE c2d)
target_compile_options(c2d_cli PRIVATE -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-parameter -Wno-implicit-fallthrough -Wno-cast-qual -Wno-sign-compare -Wno-misleading-indentation)

enable_testing()
add_test(NAME c2d_test COMMAND c2d_test)
