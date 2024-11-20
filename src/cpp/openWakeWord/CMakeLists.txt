cmake_minimum_required(VERSION 3.0)

project(openWakeWord C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

add_executable(openWakeWord
    src/openwakeword.hpp
    src/openwakeword.cpp
    src/main.cpp
)

target_link_libraries(openWakeWord PUBLIC ${SDL2_LIBRARIES} onnxruntime)

target_link_directories(openWakeWord PUBLIC
                        ${CMAKE_CURRENT_LIST_DIR}/Microsoft.ML.OnnxRuntime.1.20.0/runtimes/win-x64/native)

target_include_directories(openWakeWord PUBLIC
                           ${CMAKE_CURRENT_LIST_DIR}/Microsoft.ML.OnnxRuntime.1.20.0/build/native/include)

add_compile_definitions(openwakeword PRIVATE MODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model")

