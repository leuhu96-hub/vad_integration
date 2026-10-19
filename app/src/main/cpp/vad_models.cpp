#include "vad_models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace vad {
namespace {

Ort::SessionOptions MakeOptions(int num_threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(std::max(1, num_threads));
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

}  // namespace

std::unique_ptr<VadModel> VadModel::Create(ModelType type, Ort::Env& env,
                                           const std::vector<ModelBlob>& blobs,
                                           int num_threads) {
    switch (type) {
        case ModelType::kFireRedVad:
            if (blobs.size() < 2) throw VadError("FireRedVAD needs model + cmvn");
            return std::make_unique<FireRedVadModel>(env, blobs[0], blobs[1], num_threads);
        case ModelType::kSilero:
            return std::make_unique<SileroVadModel>(env, blobs.at(0));
        case ModelType::kYamnet:
            return std::make_unique<YamnetModel>(env, blobs.at(0), num_threads);
    }
    throw VadError("Unknown model type");
}

// ---------------------------------------------------------------- Kaldi CMVN

Cmvn ParseKaldiCmvn(const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    const uint8_t* end = p + size;
    // Optional "<key> " prefix, then binary marker "\0B".
    const uint8_t* q = p;
    while (q + 1 < end && !(q[0] == 0 && q[1] == 'B')) ++q;
    if (q + 1 >= end) throw VadError("cmvn.ark: binary marker not found");
    q += 2;
    if (q + 3 > end) throw VadError("cmvn.ark: truncated");
    const bool is_double = q[0] == 'D';
    if (!(q[0] == 'D' || q[0] == 'F') || q[1] != 'M' || q[2] != ' ') {
        throw VadError("cmvn.ark: expected DM/FM matrix");
    }
    q += 3;
    auto read_int = [&](int32_t* out) {
        if (q + 5 > end || q[0] != 4) throw VadError("cmvn.ark: bad int header");
        std::memcpy(out, q + 1, 4);
        q += 5;
    };
    int32_t rows = 0, cols = 0;
    read_int(&rows);
    read_int(&cols);
    const size_t elem = is_double ? 8 : 4;
    if (rows != 2 || cols < 2 || q + elem * rows * cols > end) throw VadError("cmvn.ark: bad shape");
    auto at = [&](int r, int c) -> double {
        const uint8_t* e = q + elem * (static_cast<size_t>(r) * cols + c);
        if (is_double) {
            double d;
            std::memcpy(&d, e, 8);
            return d;
        }
        float f;
        std::memcpy(&f, e, 4);
        return f;
    };
    const int dim = cols - 1;
    const double count = at(0, dim);
    if (count < 1) throw VadError("cmvn.ark: bad count");
    Cmvn cmvn;
    cmvn.mean.resize(dim);
    cmvn.istd.resize(dim);
    for (int d = 0; d < dim; ++d) {
        const double mean = at(0, d) / count;
        const double var = std::max(at(1, d) / count - mean * mean, 1e-20);
        cmvn.mean[d] = static_cast<float>(mean);
        cmvn.istd[d] = static_cast<float>(1.0 / std::sqrt(var));
    }
    return cmvn;
}

// ---------------------------------------------------------------- FireRedVAD

FireRedVadModel::FireRedVadModel(Ort::Env& env, ModelBlob onnx, ModelBlob cmvn, int num_threads)
    : session_(env, onnx.data, onnx.size, MakeOptions(num_threads)),
      cmvn_(ParseKaldiCmvn(cmvn.data, cmvn.size)) {
    if (static_cast<int>(cmvn_.mean.size()) != Fbank::kNumBins) throw VadError("cmvn dim mismatch");
    track_.hop_s = 0.010f;
    track_.offset_s = 0.0125f;  // centre of the 25 ms analysis window
}

void FireRedVadModel::Feed(const float* samples, size_t n) {
    const size_t old = pending_.size();
    pending_.resize(old + n);
    // Kaldi fbank works on the int16 scale.
    for (size_t i = 0; i < n; ++i) pending_[old + i] = samples[i] * 32768.0f;
    while (Fbank::NumFrames(pending_.size()) >= kChunkFrames) RunFrames(kChunkFrames);
}

void FireRedVadModel::Finish() {
    const int frames = Fbank::NumFrames(pending_.size());
    if (frames > 0) RunFrames(frames);
    pending_.clear();
}

void FireRedVadModel::RunFrames(int num_frames) {
    feat_.resize(static_cast<size_t>(num_frames) * Fbank::kNumBins);
    fbank_.Compute(pending_.data(), num_frames, feat_.data());
    cmvn_.Apply(feat_.data(), num_frames);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 3> shape{1, num_frames, Fbank::kNumBins};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, feat_.data(), feat_.size(),
                                                       shape.data(), shape.size());
    const char* in_names[] = {"feat"};
    const char* out_names[] = {"probs"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const float* probs = out[0].GetTensorData<float>();
    const size_t count = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    track_.probs.insert(track_.probs.end(), probs, probs + count);

    // The next chunk starts exactly where this chunk's next frame would have started.
    const size_t consumed = static_cast<size_t>(num_frames) * Fbank::kFrameShift;
    pending_.erase(pending_.begin(), pending_.begin() + std::min(consumed, pending_.size()));
}

// ---------------------------------------------------------------- Silero

SileroVadModel::SileroVadModel(Ort::Env& env, ModelBlob onnx)
    // The model is tiny and strictly sequential: one thread is fastest.
    : session_(env, onnx.data, onnx.size, MakeOptions(1)),
      input_(kContext + kWindow, 0.0f),
      state_(2 * 1 * 128, 0.0f) {
    track_.hop_s = static_cast<float>(kWindow) / kTargetSampleRate;
    track_.offset_s = track_.hop_s / 2;
}

void SileroVadModel::Feed(const float* samples, size_t n) {
    while (n > 0) {
        const size_t take = std::min(n, kWindow - filled_);
        std::memcpy(&input_[kContext + filled_], samples, take * sizeof(float));
        filled_ += take;
        samples += take;
        n -= take;
        if (filled_ == kWindow) RunWindow();
    }
}

void SileroVadModel::Finish() {
    if (filled_ > 0) {
        std::fill(input_.begin() + kContext + filled_, input_.end(), 0.0f);
        RunWindow();
    }
}

void SileroVadModel::RunWindow() {
    const std::array<int64_t, 2> in_shape{1, kContext + kWindow};
    const std::array<int64_t, 3> st_shape{2, 1, 128};
    std::array<Ort::Value, 3> inputs{
            Ort::Value::CreateTensor<float>(mem_, input_.data(), input_.size(), in_shape.data(), 2),
            Ort::Value::CreateTensor<float>(mem_, state_.data(), state_.size(), st_shape.data(), 3),
            Ort::Value::CreateTensor<int64_t>(mem_, &sr_, 1, nullptr, 0)};
    const char* in_names[] = {"input", "state", "sr"};
    const char* out_names[] = {"output", "stateN"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 2);
    track_.probs.push_back(out[0].GetTensorData<float>()[0]);
    std::memcpy(state_.data(), out[1].GetTensorData<float>(), state_.size() * sizeof(float));
    // The last 64 samples of this window become the context of the next one.
    std::memmove(input_.data(), input_.data() + kWindow, kContext * sizeof(float));
    filled_ = 0;
}

// ---------------------------------------------------------------- YAMNet

YamnetModel::YamnetModel(Ort::Env& env, ModelBlob onnx, int num_threads)
    : session_(env, onnx.data, onnx.size, MakeOptions(num_threads)) {
    track_.hop_s = static_cast<float>(kHop) / kTargetSampleRate;
    track_.offset_s = static_cast<float>(kPatch) / 2 / kTargetSampleRate;
}

void YamnetModel::Feed(const float* samples, size_t n) {
    pending_.insert(pending_.end(), samples, samples + n);
    total_samples_ += n;
    constexpr size_t run_len = kPatch + static_cast<size_t>(kHop) * (kFramesPerRun - 1);
    while (pending_.size() >= run_len) Run(run_len);
}

void YamnetModel::Finish() {
    // Leftover audio that is not yet covered by any emitted patch. The graph zero-pads
    // short input to one full patch; skip a sliver that the previous patch already overlaps.
    const bool nothing_yet = track_.probs.empty();
    if (!pending_.empty() && (nothing_yet || pending_.size() > kPatch - kHop)) {
        Run(pending_.size());
    }
    pending_.clear();
}

void YamnetModel::Run(size_t num_samples) {
    const std::array<int64_t, 1> shape{static_cast<int64_t>(num_samples)};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem_, pending_.data(), num_samples,
                                                       shape.data(), 1);
    const char* in_names[] = {"waveform"};
    const char* out_names[] = {"output_0"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const float* scores = out[0].GetTensorData<float>();
    const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
    const int64_t frames = dims.at(0);
    for (int64_t f = 0; f < frames; ++f) {
        const float* row = scores + f * kNumClasses;
        // AudioSet indices 0..3: Speech, Child speech, Conversation, Narration/monologue.
        track_.probs.push_back(std::max({row[0], row[1], row[2], row[3]}));
    }
    const size_t consumed = std::min(num_samples, static_cast<size_t>(frames) * kHop);
    pending_.erase(pending_.begin(), pending_.begin() + consumed);
}

}  // namespace vad
