cmake_minimum_required(VERSION 3.16)
project(dasheng-audiogen LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DASHENG_AUDIOGEN_METAL "Build with ggml-metal backend" ON)
option(DASHENG_AUDIOGEN_CUDA  "Build with ggml-cuda backend"  OFF)

if (APPLE AND DASHENG_AUDIOGEN_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if (DASHENG_AUDIOGEN_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)

add_library(dasheng_audiogen_core
    src/backend.cpp
    src/t5_encoder.cpp
    src/tokenizer.cpp
    src/content_adapter.cpp
    src/dit.cpp
    src/scheduler.cpp
    src/vocoder.cpp
    src/pipeline.cpp
)
target_include_directories(dasheng_audiogen_core PUBLIC
    src
    third_party/pocketfft
    ${SENTENCEPIECE_INCLUDE_DIRS}
)
target_link_libraries(dasheng_audiogen_core PUBLIC ggml ${SENTENCEPIECE_LIBRARIES})
target_link_directories(dasheng_audiogen_core PUBLIC ${SENTENCEPIECE_LIBRARY_DIRS})

add_executable(dasheng-audiogen src/main.cpp)
target_link_libraries(dasheng-audiogen PRIVATE dasheng_audiogen_core)
