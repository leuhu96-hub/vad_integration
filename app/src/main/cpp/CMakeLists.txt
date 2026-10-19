cmake_minimum_required(VERSION 3.22.1)
project(vadnative CXX)

if(NOT ORT_DIR)
    message(FATAL_ERROR "ORT_DIR is not set (Gradle task extractOrt unpacks onnxruntime-android)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/headers)

add_library(vadnative SHARED
        jni_bridge.cpp
        audio_decoder.cpp
        vad_models.cpp
        postprocess.cpp)

# DSP loops (resampler, fbank) are far too slow at -O0, so optimise debug builds as well.
target_compile_options(vadnative PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter -ffast-math)

target_link_libraries(vadnative
        onnxruntime
        mediandk
        android
        log)
