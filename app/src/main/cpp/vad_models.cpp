#include "vad_models.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vad {

std::unique_ptr<VadModel> VadModel::Create(ModelType type, std::vector<uint8_t> model,
                                           const std::vector<uint8_t>& extra, int num_threads) {
    switch (type) {
        case ModelType::kFireRedVad:
            if (extra.empty()) throw VadError("FireRedVAD needs cmvn.ark");
            return std::make_unique<FireRedVadModel>(CreateSession(std::move(model), num_threads), extra);
        case ModelType::kSilero:
            // The model is tiny and strictly sequential: one thread is fastest.
            return std::make_unique<SileroVadModel>(CreateSession(std::move(model), 1));
        case ModelType::kYamnet:
            return std::make_unique<YamnetModel>(CreateSession(std::move(model), num_threads));
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

FireRedVadModel::FireRedVadModel(std::unique_ptr<InferenceSession> s, const std::vector<uint8_t>& cmvn)
    : VadModel(std::move(s)), cmvn_(ParseKaldiCmvn(cmvn.data(), cmvn.size())) {
    if (static_cast<int>(cmvn_.mean.size()) != Fbank::kNumBins) throw VadError("cmvn dim mismatch");
    track_.hop_s = 0.010f;
    track_.offset_s = 0.0125f;  // centre of the 25 ms analysis window

    // Fixed input is [1, T, 80] or [1, T, 80, 1] with T > 0.
    const auto shape = session_->InputShape(0);
    if (shape.size() >= 3 && shape[1] > 0) {
        int64_t bins = 1;
        for (size_t d = 2; d < shape.size(); ++d) bins *= shape[d];
        if (bins != Fbank::kNumBins) throw VadError("FireRedVAD: unexpected feat shape");
        fixed_frames_ = static_cast<int>(shape[1]);
        fixed_shape_ = shape;
        fixed_shape_[0] = 1;
        if (fixed_frames_ <= kLookback + kLookahead) {
            // Too short for the full receptive field: keep the middle half of each window.
            LOGW("FireRedVAD: fixed T=%d < %d frames, probabilities will differ from the reference",
                 fixed_frames_, kLookback + kLookahead + 1);
            left_ctx_ = right_ctx_ = fixed_frames_ / 4;
        }
    }
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

    if (fixed_frames_ > 0) {
        RunFixed(num_frames);
    } else {
        // Some converters pad the time axis; keep exactly one probability per frame.
        const auto probs = Infer(feat_.data(), num_frames);
        track_.probs.insert(track_.probs.end(), probs.begin(),
                            probs.begin() + std::min<size_t>(probs.size(), num_frames));
    }

    // The next chunk starts exactly where this chunk's next frame would have started.
    const size_t consumed = static_cast<size_t>(num_frames) * Fbank::kFrameShift;
    pending_.erase(pending_.begin(), pending_.begin() + std::min(consumed, pending_.size()));
}

// Slides a fixed_frames_ window over feat_ (one reference chunk). A frame is taken from a
// window only where it has left_ctx_ frames before it and right_ctx_ after it inside the
// window, or where the window edge is the chunk edge itself (the reference model sees no
// context there either), so the result equals running the whole chunk at once.
void FireRedVadModel::RunFixed(int num_frames) {
    const int t = fixed_frames_;
    if (num_frames <= t) {
        // Zero (= CMVN mean) padding is not the same as the graph's own edge padding:
        // the last few frames of a chunk shorter than T differ slightly.
        window_.assign(static_cast<size_t>(t) * Fbank::kNumBins, 0.0f);
        std::copy(feat_.begin(), feat_.begin() + static_cast<size_t>(num_frames) * Fbank::kNumBins,
                  window_.begin());
        const auto probs = Infer(window_.data(), t);
        track_.probs.insert(track_.probs.end(), probs.begin(), probs.begin() + num_frames);
        return;
    }
    for (int done = 0; done < num_frames;) {
        const int start = std::min(std::max(0, done - left_ctx_), num_frames - t);
        const int end = start + t == num_frames ? num_frames : start + t - right_ctx_;
        const auto probs = Infer(feat_.data() + static_cast<size_t>(start) * Fbank::kNumBins, t);
        track_.probs.insert(track_.probs.end(), probs.begin() + (done - start), probs.begin() + (end - start));
        done = end;
    }
}

std::vector<float> FireRedVadModel::Infer(const float* feat, int num_frames) {
    const std::vector<int64_t> shape =
            fixed_frames_ > 0 ? fixed_shape_ : std::vector<int64_t>{1, num_frames, Fbank::kNumBins};
    auto out = session_->Run({{"feat", DType::kFloat32, shape, feat}}, {"probs"});
    if (fixed_frames_ > 0 && out[0].data.size() < static_cast<size_t>(num_frames)) {
        throw VadError("FireRedVAD: model returned fewer probabilities than frames");
    }
    return std::move(out[0].data);
}

// ---------------------------------------------------------------- Silero

SileroVadModel::SileroVadModel(std::unique_ptr<InferenceSession> s)
    : VadModel(std::move(s)),
      has_sr_(session_->HasInput("sr")),
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
    std::vector<InputTensor> inputs{
            {"input", DType::kFloat32, {1, kContext + kWindow}, input_.data()},
            {"state", DType::kFloat32, {2, 1, 128}, state_.data()}};
    // The stock graph takes the sample rate; the TFLite export has it folded in.
    if (has_sr_) inputs.push_back({"sr", DType::kInt64, {}, &sr_});
    auto out = session_->Run(inputs, {"output", "stateN"});
    track_.probs.push_back(out[0].data.at(0));
    if (out[1].data.size() != state_.size()) throw VadError("Silero: unexpected state size");
    std::memcpy(state_.data(), out[1].data.data(), state_.size() * sizeof(float));
    // The last 64 samples of this window become the context of the next one.
    std::memmove(input_.data(), input_.data() + kWindow, kContext * sizeof(float));
    filled_ = 0;
}

// ---------------------------------------------------------------- YAMNet

YamnetModel::YamnetModel(std::unique_ptr<InferenceSession> s) : VadModel(std::move(s)) {
    // Fixed input is [N] or [1, N]; the retrained model takes one 0.96 s window (N = 15360).
    const auto shape = session_->InputShape(0);
    if (!shape.empty() && shape.size() <= 2 && shape.back() > 0) {
        fixed_len_ = static_cast<size_t>(shape.back());
        batched_ = shape.size() == 2;
    }
    const size_t patch = fixed_len_ > 0 ? fixed_len_ : static_cast<size_t>(kPatch);
    track_.hop_s = static_cast<float>(fixed_len_ > 0 ? kHop : kGraphHop) / kTargetSampleRate;
    track_.offset_s = static_cast<float>(patch) / 2 / kTargetSampleRate;
}

void YamnetModel::Feed(const float* samples, size_t n) {
    pending_.insert(pending_.end(), samples, samples + n);
    total_samples_ += n;
    if (fixed_len_ > 0) {
        // Patch k covers samples [k * hop, k * hop + fixed_len).
        while (pending_start_ + pending_.size() >= next_frame_ * kHop + fixed_len_) {
            RunFixed(pending_.data() + (next_frame_ * kHop - pending_start_));
            ++next_frame_;
        }
        const size_t drop = std::min(pending_.size(), next_frame_ * kHop - pending_start_);
        pending_.erase(pending_.begin(), pending_.begin() + drop);
        pending_start_ += drop;
        return;
    }
    constexpr size_t run_len = kPatch + static_cast<size_t>(kGraphHop) * (kFramesPerRun - 1);
    while (pending_.size() >= run_len) RunDynamic(run_len);
}

void YamnetModel::Finish() {
    FlushWindows();
    ToSlots();
}

void YamnetModel::ToSlots() {
    // Window i covers [i * hop, i * hop + 2 * offset) with its score in probs[i]. Cut the
    // timeline into hop-sized slots; a slot's probability is the average of the windows
    // covering it, weighted by a triangle that peaks at the window centre and falls to 0 at
    // its edges.
    const float hop = track_.hop_s;
    const float half = track_.offset_s;
    const int n = static_cast<int>(track_.probs.size());
    if (n == 0) return;
    const float duration_s = static_cast<float>(total_samples_) / kTargetSampleRate;
    const int slots = std::max(1, static_cast<int>(std::ceil(duration_s / hop - 1e-4f)));
    std::vector<float> num(slots, 0.0f), den(slots, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float centre = half + i * hop;
        const int k0 = std::max(0, static_cast<int>(std::floor((centre - half) / hop)));
        const int k1 = std::min(slots - 1, static_cast<int>(std::ceil((centre + half) / hop)));
        for (int k = k0; k <= k1; ++k) {
            const float w = 1.0f - std::fabs((k + 0.5f) * hop - centre) / half;
            if (w <= 0) continue;
            num[k] += w * track_.probs[i];
            den[k] += w;
        }
    }
    std::vector<float> slot_prob(slots);
    for (int k = 0; k < slots; ++k) {
        if (den[k] > 0) {
            slot_prob[k] = num[k] / den[k];
        } else {
            // No window reaches this slot: take the nearest one.
            const int i = std::clamp(static_cast<int>(((k + 0.5f) * hop - half) / hop + 0.5f), 0, n - 1);
            slot_prob[k] = track_.probs[i];
        }
    }
    track_.probs = std::move(slot_prob);
    track_.offset_s = hop / 2;  // probs[k] is now centred at (k + 0.5) * hop
}

void YamnetModel::FlushWindows() {
    if (fixed_len_ > 0) {
        // Patches cover [k * hop, k * hop + fixed_len): 1 + ceil((N - fixed_len) / hop)
        // of them, zero-padding the tail.
        const size_t target = total_samples_ <= fixed_len_
                ? 1 : 1 + (total_samples_ - fixed_len_ + kHop - 1) / kHop;
        std::vector<float> window(fixed_len_);
        while (next_frame_ < target) {
            std::fill(window.begin(), window.end(), 0.0f);
            const size_t start = next_frame_ * kHop - pending_start_;
            if (start < pending_.size()) {
                const size_t n = std::min(fixed_len_, pending_.size() - start);
                std::copy_n(pending_.begin() + start, n, window.begin());
            }
            RunFixed(window.data());
            ++next_frame_;
        }
        pending_.clear();
        return;
    }
    // Leftover audio that is not yet covered by any emitted patch. The graph zero-pads
    // short input to one full patch; skip a sliver that the previous patch already overlaps.
    const bool nothing_yet = track_.probs.empty();
    if (!pending_.empty() && (nothing_yet || pending_.size() > kPatch - kGraphHop)) {
        RunDynamic(pending_.size());
    }
    pending_.clear();
}

void YamnetModel::RunFixed(const float* window) {
    std::vector<int64_t> shape{static_cast<int64_t>(fixed_len_)};
    if (batched_) shape.insert(shape.begin(), 1);
    auto out = session_->Run({{"waveform", DType::kFloat32, shape, window}}, {"output_0"});
    AppendScores(out[0]);
}

void YamnetModel::RunDynamic(size_t num_samples) {
    auto out = session_->Run({{"waveform", DType::kFloat32, {static_cast<int64_t>(num_samples)}, pending_.data()}},
                             {"output_0"});
    const size_t frames = out[0].data.size() / kNumClasses;
    AppendScores(out[0]);
    const size_t consumed = std::min(num_samples, frames * kGraphHop);
    pending_.erase(pending_.begin(), pending_.begin() + consumed);
}

void YamnetModel::AppendScores(const OutputTensor& scores) {
    const auto& d = scores.data;
    if (fixed_len_ > 0 && d.size() <= 2) {
        // Retrained model: one score per 0.96 s window.
        //  1 value  -> speech probability (sigmoid applied if it looks like a logit);
        //  2 values -> [non-speech, speech], softmax applied if they are not already probabilities.
        float p;
        if (d.size() == 1) {
            p = (d[0] >= 0.0f && d[0] <= 1.0f) ? d[0] : 1.0f / (1.0f + std::exp(-d[0]));
        } else {
            const bool is_prob = d[0] >= 0.0f && d[1] >= 0.0f && std::fabs(d[0] + d[1] - 1.0f) < 1e-3f;
            p = is_prob ? d[1] : 1.0f / (1.0f + std::exp(d[0] - d[1]));
        }
        track_.probs.push_back(p);
        return;
    }
    if (scores.data.size() % kNumClasses != 0) throw VadError("YAMNet: expected 1, 2 or 521 scores");
    for (size_t f = 0; f < scores.data.size() / kNumClasses; ++f) {
        const float* row = scores.data.data() + f * kNumClasses;
        // AudioSet indices 0..3: Speech, Child speech, Conversation, Narration/monologue.
        track_.probs.push_back(std::max({row[0], row[1], row[2], row[3]}));
    }
}

}  // namespace vad
