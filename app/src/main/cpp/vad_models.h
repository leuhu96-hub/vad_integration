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
// I/O: feat [1, T, 80] -> probs [1, T, 1]. T may be dynamic or fixed (NPU exports, possibly
// with a trailing [.., 80, 1]); a fixed T is run as overlapping windows so that every kept
// frame sees its full receptive field and matches the dynamic model.
class FireRedVadModel : public VadModel {
public:
    FireRedVadModel(std::unique_ptr<InferenceSession> s, const std::vector<uint8_t>& cmvn);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

private:
    // Same chunking as the reference implementation (chunk_max_frame = 30000, i.e. 300 s).
    static constexpr int kChunkFrames = 30000;
    // Receptive field of the bundled DFSMN: 8 FSMN layers, each with a 20-tap lookback
    // (t-19..t) and a 20-tap lookahead (t+1..t+20) filter.
    static constexpr int kLookback = 8 * 19;
    static constexpr int kLookahead = 8 * 20;
    void RunFrames(int num_frames);
    void RunFixed(int num_frames);
    std::vector<float> Infer(const float* feat, int num_frames);

    Fbank fbank_;
    Cmvn cmvn_;
    std::vector<float> pending_;  // int16-scaled samples not yet consumed
    std::vector<float> feat_;
    int fixed_frames_ = 0;              // >0: graph takes exactly this many frames
    std::vector<int64_t> fixed_shape_;  // declared input shape, batch forced to 1
    int left_ctx_ = kLookback, right_ctx_ = kLookahead;
    std::vector<float> window_;  // zero-padded input when a chunk is shorter than fixed_frames_
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

// After Finish(), track() holds one probability per hop-sized slot (not per window): each slot
// is the triangle-weighted average of the windows covering it, centred at (k + 0.5) * hop.
//
// YAMNet-based speech detector: a 0.96 s window (15360 samples, mono 16 kHz) slides with a
// 0.08 s hop; each window yields one speech probability (see AppendScores for the accepted
// output layouts). Legacy graphs are still supported:
//  - dynamic waveform [N] -> scores [frames, 521] (tf2onnx export): run on long chunks;
//    the graph itself frames the audio with a fixed 0.48 s hop (kGraphHop), speech
//    probability is the max over Speech/Child speech/Conversation/Narration;
//  - fixed waveform [N] -> scores [1, 521] (TF-Hub TFLite): same sliding scheme as above.
class YamnetModel : public VadModel {
public:
    explicit YamnetModel(std::unique_ptr<InferenceSession> s);
    void Feed(const float* samples, size_t n) override;
    void Finish() override;

    static constexpr int kPatch = 15360;  // 0.96 s of spectrogram frames
    static constexpr int kHop = 1280;     // 0.08 s (fixed-input graphs)
    static constexpr int kGraphHop = 7680;  // 0.48 s (hop built into whole-waveform graphs)

private:
    static constexpr int kFramesPerRun = 32;
    static constexpr int kNumClasses = 521;
    void FlushWindows();  // emits the trailing windows (per-window scores)
    void ToSlots();       // per-window scores -> per-slot probabilities (triangle-weighted)
    void RunDynamic(size_t num_samples);
    void RunFixed(const float* window);
    void AppendScores(const OutputTensor& scores);

    size_t fixed_len_ = 0;    // >0: graph takes exactly this many samples
    bool batched_ = false;    // fixed input is [1, N] rather than [N]
    std::vector<float> pending_;
    size_t pending_start_ = 0;  // absolute sample index of pending_[0]
    size_t total_samples_ = 0;
    size_t next_frame_ = 0;     // fixed mode: next patch to emit
};

Cmvn ParseKaldiCmvn(const void* data, size_t size);

}  // namespace vad
