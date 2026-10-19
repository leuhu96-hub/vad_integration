cmake_minimum_required(VERSION 3.22.1)
project(vadnative CXX)

if(NOT ORT_DIR OR NOT TFLITE_DIR)
    message(FATAL_ERROR "ORT_DIR/TFLITE_DIR not set (Gradle tasks extractOrt/extractTflite unpack the AARs)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/headers)

# LiteRT's JNI library also exports the TFLite C API (TfLiteInterpreter*).
add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES
        IMPORTED_LOCATION ${TFLITE_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so
        INTERFACE_INCLUDE_DIRECTORIES ${TFLITE_DIR}/headers)

add_library(vadnative SHARED
        jni_bridge.cpp
        audio_decoder.cpp
        inference.cpp
        vad_models.cpp
        postprocess.cpp)

# DSP loops (resampler, fbank) are far too slow at -O0, so optimise debug builds as well.
target_compile_options(vadnative PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter -ffast-math)

target_link_libraries(vadnative
        onnxruntime
        tflite
        mediandk
        android
        log)
