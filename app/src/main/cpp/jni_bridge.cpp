#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>

#include <chrono>
#include <memory>
#include <vector>

#include "audio_decoder.h"
#include "common.h"
#include "postprocess.h"
#include "vad_models.h"

using namespace vad;

namespace {

Ort::Env& GetEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vadnative");
    return env;
}

struct AssetCloser {
    void operator()(AAsset* a) const { AAsset_close(a); }
};
using AssetPtr = std::unique_ptr<AAsset, AssetCloser>;

AssetPtr OpenAsset(AAssetManager* mgr, const char* path, ModelBlob* blob) {
    AssetPtr asset(AAssetManager_open(mgr, path, AASSET_MODE_BUFFER));
    if (!asset) throw VadError(std::string("Missing asset ") + path);
    blob->data = AAsset_getBuffer(asset.get());
    blob->size = static_cast<size_t>(AAsset_getLength(asset.get()));
    if (!blob->data) throw VadError(std::string("Cannot read asset ") + path);
    return asset;
}

std::vector<const char*> AssetPaths(ModelType type) {
    switch (type) {
        case ModelType::kFireRedVad: return {"models/fireredvad_vad.onnx", "models/cmvn.ark"};
        case ModelType::kSilero: return {"models/silero_vad.onnx"};
        case ModelType::kYamnet: return {"models/yamnet.onnx"};
    }
    throw VadError("Unknown model type");
}

void ThrowJava(JNIEnv* env, const char* cls, const char* msg) {
    if (env->ExceptionCheck()) return;
    jclass c = env->FindClass(cls);
    if (c) env->ThrowNew(c, msg);
}

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_vadapp_vad_NativeVad_analyze(JNIEnv* env, jobject /*thiz*/, jobject asset_manager,
                                              jint model_type, jint fd, jlong offset, jlong length,
                                              jint num_threads, jobject listener) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto type = static_cast<ModelType>(model_type);

    jclass listener_cls = env->GetObjectClass(listener);
    jmethodID on_progress = env->GetMethodID(listener_cls, "onProgress", "(F)Z");

    try {
        AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
        std::vector<AssetPtr> assets;
        std::vector<ModelBlob> blobs;
        for (const char* path : AssetPaths(type)) {
            ModelBlob blob;
            assets.push_back(OpenAsset(mgr, path, &blob));
            blobs.push_back(blob);
        }
        auto model = VadModel::Create(type, GetEnv(), blobs, num_threads);
        assets.clear();  // ORT keeps its own copy of the graph

        size_t total_samples = 0;
        std::chrono::steady_clock::duration model_time{};
        PcmSink sink = [&](const float* s, size_t n) {
            total_samples += n;
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
        LOGI("model %d: %zu probs, %.2fs audio, %lld ms total, %lld ms in model", model_type,
             track.probs.size(), duration, static_cast<long long>(elapsed_ms),
             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(model_time).count()));

        jfloatArray probs = env->NewFloatArray(static_cast<jsize>(track.probs.size()));
        env->SetFloatArrayRegion(probs, 0, static_cast<jsize>(track.probs.size()), track.probs.data());
        jstring mime = env->NewStringUTF(info.mime.c_str());
        jclass result_cls = env->FindClass("com/example/vadapp/vad/AnalysisResult");
        jmethodID ctor = env->GetMethodID(result_cls, "<init>", "([FFFFIILjava/lang/String;J)V");
        return env->NewObject(result_cls, ctor, probs, track.hop_s, track.offset_s, duration,
                              info.sample_rate, info.channels, mime,
                              static_cast<jlong>(elapsed_ms));
    } catch (const Cancelled&) {
        ThrowJava(env, "java/util/concurrent/CancellationException", "Analysis cancelled");
    } catch (const Ort::Exception& e) {
        LOGE("onnxruntime: %s", e.what());
        ThrowJava(env, "java/lang/RuntimeException", (std::string("ONNX Runtime: ") + e.what()).c_str());
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
                                              jint pad_ms) {
    ProbTrack track;
    track.hop_s = hop_s;
    track.offset_s = offset_s;
    track.probs.resize(env->GetArrayLength(probs));
    env->GetFloatArrayRegion(probs, 0, static_cast<jsize>(track.probs.size()), track.probs.data());

    SegmentParams params;
    params.threshold = threshold;
    params.min_speech_ms = min_speech_ms;
    params.min_silence_ms = min_silence_ms;
    params.pad_ms = pad_ms;

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
