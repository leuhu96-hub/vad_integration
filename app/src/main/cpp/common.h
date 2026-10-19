#pragma once

#include <android/log.h>

#include <stdexcept>
#include <string>

#define VAD_TAG "VadNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, VAD_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, VAD_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, VAD_TAG, __VA_ARGS__)

namespace vad {

// All models consume 16 kHz mono float audio in [-1, 1].
constexpr int kTargetSampleRate = 16000;

// Must match ModelType in Kotlin.
enum class ModelType : int {
    kFireRedVad = 0,
    kSilero = 1,
    kYamnet = 2,
};

struct VadError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when the progress callback asks to stop.
struct Cancelled : std::exception {};

}  // namespace vad
