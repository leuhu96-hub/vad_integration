#pragma once

#include <memory>
#include <vector>

#include "common.h"
#include "fbank.h"
#include "inference.h"

namespace vad {

// Frame-level speech probability track produced by a model.
// probs[i] describes audio centred at offset_s + i * hop_s.
struct ProbTrack {
    std::vector<float> probs;
    float hop_s = 0;
    float offset_s = 0;
};

// Incremental model runner: audio is pushed as it is decoded so a long video
// never has to be held in memory as a whole. The underlying runtime (ONNX Runtime
// or TFLite) is chosen from the model file itself.
class VadModel {
public:
    virtual ~VadModel() = default;
    // 16 kHz mono float samples in [-1, 1].
    virtual void Feed(const float* samples, size_t n) = 0;
    virtual void Finish() = 0;
    const ProbTrack& track() const { return track_; }
    ModelFormat format() const { return session_->format(); }

    // `extra` is the CMVN statistics for FireRedVAD and unused otherwise.
    static std::unique_ptr<VadModel> Create(ModelType type, std::vector<uint8_t> model,
                                            const std::vector<uint8_t>& extra, int num_threads);

protected:
    explicit VadModel(std::unique_ptr<InferenceSession> s) : session_(std::move(s)) {}
    std::unique_ptr<InferenceSession> session_;
    ProbTrack track_;
};

// FireRedVAD non-streaming DFSMN: 80-dim fbank + CMVN -> per-10ms speech probability.
// I/O: feat [1, T, 80] -> probs [1, T, 1].
class FireRedVadModel : public VadModel {
public:
    FireRedVadModel(std::unique_ptr<InferenceSession> s, const std::vector<uint8_t>& cmvn);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

private:
    // Same chunking as the reference implementation (chunk_max_frame = 30000, i.e. 300 s).
    static constexpr int kChunkFrames = 30000;
    void RunFrames(int num_frames);

    Fbank fbank_;
    Cmvn cmvn_;
    std::vector<float> pending_;  // int16-scaled samples not yet consumed
    std::vector<float> feat_;
};

// Silero VAD v5/v6: 512-sample windows with 64 samples of context and a recurrent state.
// I/O: input [1, 576], state [2, 1, 128] (+ sr scalar if the graph has it)
//      -> output [1, 1], stateN [2, 1, 128].
class SileroVadModel : public VadModel {
public:
    explicit SileroVadModel(std::unique_ptr<InferenceSession> s);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

private:
    static constexpr int kWindow = 512;
    static constexpr int kContext = 64;
    void RunWindow();

    bool has_sr_;
    std::vector<float> input_;  // [context | window]
    size_t filled_ = 0;         // samples of the current window already in input_
    std::vector<float> state_;  // [2, 1, 128]
    int64_t sr_ = kTargetSampleRate;
};

// YAMNet (AudioSet, 521 classes): 0.96 s patches every 0.48 s. Speech probability is the
// max over the speech classes (Speech, Child speech, Conversation, Narration).
// Two graph flavours are supported:
//  - dynamic waveform [N] -> scores [frames, 521] (tf2onnx export): run on long chunks;
//  - fixed waveform [15600] -> scores [1, 521] (TF-Hub TFLite): slide one patch at a time.
class YamnetModel : public VadModel {
public:
    explicit YamnetModel(std::unique_ptr<InferenceSession> s);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

    static constexpr int kPatch = 15360;  // 0.96 s of spectrogram frames
    static constexpr int kHop = 7680;     // 0.48 s

private:
    static constexpr int kFramesPerRun = 32;
    static constexpr int kNumClasses = 521;
    void RunDynamic(size_t num_samples);
    void RunFixed(const float* window);
    void AppendScores(const OutputTensor& scores);

    size_t fixed_len_ = 0;    // >0: graph takes exactly this many samples
    std::vector<float> pending_;
    size_t pending_start_ = 0;  // absolute sample index of pending_[0]
    size_t total_samples_ = 0;
    size_t next_frame_ = 0;     // fixed mode: next patch to emit
};

Cmvn ParseKaldiCmvn(const void* data, size_t size);

}  // namespace vad
