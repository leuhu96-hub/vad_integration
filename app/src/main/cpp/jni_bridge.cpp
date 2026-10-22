#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "audio_decoder.h"
#include "common.h"
#include "postprocess.h"
#include "vad_models.h"

using namespace vad;

namespace {

constexpr char kAssetPrefix[] = "asset:";

// Loads "asset:<path inside assets>" or an absolute file path.
std::vector<uint8_t> ReadModelFile(AAssetManager* mgr, const std::string& path) {
    std::vector<uint8_t> bytes;
    if (path.rfind(kAssetPrefix, 0) == 0) {
        const std::string name = path.substr(sizeof(kAssetPrefix) - 1);
        AAsset* asset = AAssetManager_open(mgr, name.c_str(), AASSET_MODE_BUFFER);
        if (!asset) throw VadError("Missing asset " + name);
        const auto* data = static_cast<const uint8_t*>(AAsset_getBuffer(asset));
        const size_t size = static_cast<size_t>(AAsset_getLength(asset));
        if (data) bytes.assign(data, data + size);
        AAsset_close(asset);
        if (!data) throw VadError("Cannot read asset " + name);
    } else {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw VadError("Cannot open model file " + path);
        std::fseek(f, 0, SEEK_END);
        bytes.resize(static_cast<size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        const size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        if (got != bytes.size()) throw VadError("Cannot read model file " + path);
    }
    return bytes;
}

std::string JString(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

void ThrowJava(JNIEnv* env, const char* cls, const char* msg) {
    if (env->ExceptionCheck()) return;
    jclass c = env->FindClass(cls);
    if (c) env->ThrowNew(c, msg);
}

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_vadapp_vad_NativeVad_analyze(JNIEnv* env, jobject /*thiz*/, jobject asset_manager,
                                              jint model_type, jstring model_path, jstring extra_path,
                                              jint fd, jlong offset, jlong length,
                                              jint num_threads, jobject listener) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto type = static_cast<ModelType>(model_type);

    jclass listener_cls = env->GetObjectClass(listener);
    jmethodID on_progress = env->GetMethodID(listener_cls, "onProgress", "(F)Z");

    try {
        AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
        const std::string extra = JString(env, extra_path);
        auto model = VadModel::Create(type, ReadModelFile(mgr, JString(env, model_path)),
                                      extra.empty() ? std::vector<uint8_t>{} : ReadModelFile(mgr, extra),
                                      num_threads);
        const char* format = FormatName(model->format());

        size_t total_samples = 0;
        std::chrono::steady_clock::duration model_time{};
        // Peak envelope of the 16 kHz mono signal for the waveform view, one value per bucket.
        constexpr size_t kEnvelopeBucket = kTargetSampleRate / 100;  // 10 ms
        std::vector<float> envelope;
        float bucket_peak = 0.0f;
        size_t bucket_fill = 0;
        PcmSink sink = [&](const float* s, size_t n) {
            total_samples += n;
            for (size_t i = 0; i < n; ++i) {
                bucket_peak = std::max(bucket_peak, std::fabs(s[i]));
                if (++bucket_fill == kEnvelopeBucket) {
                    envelope.push_back(bucket_peak);
                    bucket_peak = 0.0f;
                    bucket_fill = 0;
                }
            }
            const auto m0 = std::chrono::steady_clock::now();
            model->Feed(s, n);
            model_time += std::chrono::steady_clock::now() - m0;
        };
        ProgressFn progress = [&](float p) {
            const jboolean keep_going = env->CallBooleanMethod(listener, on_progress, p);
            return !env->ExceptionCheck() && keep_going;
        };
        AudioInfo info = DecodeToMono16k(fd, offset, length, sink, progress);
        const auto m0 = std::chrono::steady_clock::now();
        model->Finish();
        model_time += std::chrono::steady_clock::now() - m0;
        if (!progress(1.0f)) throw Cancelled();

        const ProbTrack& track = model->track();
        const float duration = static_cast<float>(total_samples) / kTargetSampleRate;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
        LOGI("model %d (%s): %zu probs, %.2fs audio, %lld ms total, %lld ms in model", model_type,
             format, track.probs.size(), duration, static_cast<long long>(elapsed_ms),
             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(model_time).count()));

        jfloatArray probs = env->NewFloatArray(static_cast<jsize>(track.probs.size()));
        env->SetFloatArrayRegion(probs, 0, static_cast<jsize>(track.probs.size()), track.probs.data());
        if (bucket_fill > 0) envelope.push_back(bucket_peak);
        const float peak = envelope.empty() ? 0.0f : *std::max_element(envelope.begin(), envelope.end());
        if (peak > 0.0f) {
            for (float& v : envelope) v /= peak;  // normalise so quiet files stay readable
        }
        jfloatArray env_arr = env->NewFloatArray(static_cast<jsize>(envelope.size()));
        env->SetFloatArrayRegion(env_arr, 0, static_cast<jsize>(envelope.size()), envelope.data());
        jstring mime = env->NewStringUTF(info.mime.c_str());
        jclass result_cls = env->FindClass("com/example/vadapp/vad/AnalysisResult");
        jmethodID ctor = env->GetMethodID(result_cls, "<init>", "([FFFFIILjava/lang/String;Ljava/lang/String;J[FF)V");
        return env->NewObject(result_cls, ctor, probs, track.hop_s, track.offset_s, duration,
                              info.sample_rate, info.channels, mime, env->NewStringUTF(format),
                              static_cast<jlong>(elapsed_ms), env_arr,
                              static_cast<float>(kEnvelopeBucket) / kTargetSampleRate);
    } catch (const Cancelled&) {
        ThrowJava(env, "java/util/concurrent/CancellationException", "Analysis cancelled");
    } catch (const std::exception& e) {
        LOGE("%s", e.what());
        ThrowJava(env, "java/lang/RuntimeException", e.what());
    }
    return nullptr;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_vadapp_vad_NativeVad_segment(JNIEnv* env, jobject /*thiz*/, jint model_type,
                                              jfloatArray probs, jfloat hop_s, jfloat offset_s,
                                              jfloat duration_s, jfloat threshold,
                                              jint min_speech_ms, jint min_silence_ms,
                                              jint pre_pad_ms, jint post_pad_ms) {
    ProbTrack track;
    track.hop_s = hop_s;
    track.offset_s = offset_s;
    track.probs.resize(env->GetArrayLength(probs));
    env->GetFloatArrayRegion(probs, 0, static_cast<jsize>(track.probs.size()), track.probs.data());

    SegmentParams params;
    params.threshold = threshold;
    params.min_speech_ms = min_speech_ms;
    params.min_silence_ms = min_silence_ms;
    params.pre_pad_ms = pre_pad_ms;
    params.post_pad_ms = post_pad_ms;

    const auto segs = MakeSegments(static_cast<ModelType>(model_type), track, duration_s, params);
    std::vector<float> flat;
    flat.reserve(segs.size() * 2);
    for (const auto& s : segs) {
        flat.push_back(s.first);
        flat.push_back(s.second);
    }
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(flat.size()));
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(flat.size()), flat.data());
    return out;
}
