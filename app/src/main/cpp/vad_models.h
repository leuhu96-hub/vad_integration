#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

#include "common.h"
#include "fbank.h"

namespace vad {

struct ModelBlob {
    const void* data = nullptr;
    size_t size = 0;
};

// Frame-level speech probability track produced by a model.
// probs[i] describes audio centred at offset_s + i * hop_s.
struct ProbTrack {
    std::vector<float> probs;
    float hop_s = 0;
    float offset_s = 0;
};

// Incremental model runner: audio is pushed as it is decoded so a long video
// never has to be held in memory as a whole.
class VadModel {
public:
    virtual ~VadModel() = default;
    // 16 kHz mono float samples in [-1, 1].
    virtual void Feed(const float* samples, size_t n) = 0;
    virtual void Finish() = 0;
    const ProbTrack& track() const { return track_; }

    static std::unique_ptr<VadModel> Create(ModelType type, Ort::Env& env,
                                            const std::vector<ModelBlob>& blobs,
                                            int num_threads);

protected:
    ProbTrack track_;
};

// FireRedVAD non-streaming DFSMN: 80-dim fbank + CMVN -> per-10ms speech probability.
class FireRedVadModel : public VadModel {
public:
    FireRedVadModel(Ort::Env& env, ModelBlob onnx, ModelBlob cmvn, int num_threads);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

private:
    // Same chunking as the reference implementation (chunk_max_frame = 30000, i.e. 300 s).
    static constexpr int kChunkFrames = 30000;
    void RunFrames(int num_frames);

    Ort::Session session_{nullptr};
    Fbank fbank_;
    Cmvn cmvn_;
    std::vector<float> pending_;  // int16-scaled samples not yet consumed
    std::vector<float> feat_;
};

// Silero VAD v5/v6: 512-sample windows with 64 samples of context and a recurrent state.
class SileroVadModel : public VadModel {
public:
    SileroVadModel(Ort::Env& env, ModelBlob onnx);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

private:
    static constexpr int kWindow = 512;
    static constexpr int kContext = 64;
    void RunWindow();

    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::vector<float> input_;  // [context | window]
    size_t filled_ = 0;         // samples of the current window already in input_
    std::vector<float> state_;  // [2, 1, 128]
    int64_t sr_ = kTargetSampleRate;
};

// YAMNet (AudioSet, 521 classes): 0.96 s patches every 0.48 s. Speech probability is the
// max over the speech classes (Speech, Child speech, Conversation, Narration).
class YamnetModel : public VadModel {
public:
    YamnetModel(Ort::Env& env, ModelBlob onnx, int num_threads);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

    static constexpr int kPatch = 15360;  // 0.96 s
    static constexpr int kHop = 7680;     // 0.48 s

private:
    static constexpr int kFramesPerRun = 32;
    static constexpr int kNumClasses = 521;
    void Run(size_t num_samples);

    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::vector<float> pending_;
    size_t total_samples_ = 0;
};

Cmvn ParseKaldiCmvn(const void* data, size_t size);

}  // namespace vad
