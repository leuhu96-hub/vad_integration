#include "postprocess.h"

#include <algorithm>
#include <cmath>

namespace vad {
namespace {

// ------------------------------------------------- FireRedVAD (vad_postprocessor.py)

constexpr float kFrameShiftS = 0.010f;
constexpr float kFrameLengthS = 0.025f;

class FireRedPostprocessor {
public:
    FireRedPostprocessor(int smooth_window, float threshold, int min_speech, int max_speech,
                         int min_silence, int extend_pre, int extend_post)
        : smooth_(std::max(1, smooth_window)), threshold_(threshold), min_speech_(min_speech),
          max_speech_(max_speech), min_silence_(min_silence), extend_pre_(extend_pre),
          extend_post_(extend_post) {}

    std::vector<int> Process(const std::vector<float>& probs) const {
        if (probs.empty()) return {};
        std::vector<float> smoothed = Smooth(probs);
        std::vector<int> binary(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) binary[i] = smoothed[i] >= threshold_;
        std::vector<int> d = StateMachine(binary);
        d = FixSmoothWindowStart(d);
        d = Extend(d);
        return SplitLong(d, probs);
    }

    std::vector<Segment> ToSegments(const std::vector<int>& d, float wav_dur) const {
        std::vector<Segment> segs;
        int start = -1;
        for (int t = 0; t < static_cast<int>(d.size()); ++t) {
            if (d[t] == 1 && start < 0) {
                start = t;
            } else if (d[t] == 0 && start >= 0) {
                segs.emplace_back(start * kFrameShiftS, t * kFrameShiftS);
                start = -1;
            }
        }
        if (start >= 0) {
            float end = d.size() * kFrameShiftS + kFrameLengthS;
            if (wav_dur > 0) end = std::min(end, wav_dur);
            segs.emplace_back(start * kFrameShiftS, end);
        }
        return segs;
    }

private:
    // Trailing moving average; the first frames use a cumulative mean.
    std::vector<float> Smooth(const std::vector<float>& p) const {
        if (smooth_ <= 1) return p;
        std::vector<float> out(p.size());
        double sum = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            sum += p[i];
            if (i >= static_cast<size_t>(smooth_)) sum -= p[i - smooth_];
            const size_t n = std::min(i + 1, static_cast<size_t>(smooth_));
            out[i] = static_cast<float>(sum / n);
        }
        return out;
    }

    std::vector<int> StateMachine(const std::vector<int>& b) const {
        if (min_speech_ <= 0 && min_silence_ <= 0) return b;
        enum { kSilence, kPossibleSpeech, kSpeech, kPossibleSilence } state = kSilence;
        std::vector<int> d(b.size(), 0);
        int speech_start = -1, silence_start = -1;
        for (int t = 0; t < static_cast<int>(b.size()); ++t) {
            const bool s = b[t];
            switch (state) {
                case kSilence:
                    if (s) { state = kPossibleSpeech; speech_start = t; }
                    break;
                case kPossibleSpeech:
                    if (s) {
                        if (t - speech_start >= min_speech_) {
                            state = kSpeech;
                            std::fill(d.begin() + speech_start, d.begin() + t, 1);
                        }
                    } else {
                        state = kSilence;
                        speech_start = -1;
                    }
                    break;
                case kSpeech:
                    if (!s) { state = kPossibleSilence; silence_start = t; }
                    break;
                case kPossibleSilence:
                    if (!s) {
                        if (t - silence_start >= min_silence_) { state = kSilence; speech_start = -1; }
                    } else {
                        state = kSpeech;
                        silence_start = -1;
                    }
                    break;
            }
            d[t] = (state == kSpeech || state == kPossibleSilence) ? 1 : 0;
        }
        return d;
    }

    std::vector<int> FixSmoothWindowStart(const std::vector<int>& d) const {
        std::vector<int> out = d;
        for (int t = 1; t < static_cast<int>(d.size()); ++t) {
            if (d[t - 1] == 0 && d[t] == 1) {
                std::fill(out.begin() + std::max(0, t - smooth_), out.begin() + t, 1);
            }
        }
        return out;
    }

    std::vector<int> Extend(const std::vector<int>& d) const {
        if (extend_pre_ <= 0 && extend_post_ <= 0) return d;
        const int n = static_cast<int>(d.size());
        std::vector<int> out(n, 0);
        for (int t = 0; t < n; ++t) {
            if (d[t] == 1) {
                std::fill(out.begin() + std::max(0, t - extend_pre_), out.begin() + std::min(n, t + extend_post_ + 1), 1);
            }
        }
        return out;
    }

    std::vector<int> SplitLong(const std::vector<int>& d, const std::vector<float>& probs) const {
        std::vector<int> out = d;
        for (const auto& seg : ToSegments(d, 0)) {
            const int sf = static_cast<int>(std::lround(seg.first / kFrameShiftS));
            const int ef = std::min(static_cast<int>(std::lround(seg.second / kFrameShiftS)),
                                    static_cast<int>(probs.size()));
            if (ef - sf <= max_speech_) continue;
            int start = 0;
            const int len = ef - sf;
            while (len - start > max_speech_) {
                const int ws = start + max_speech_ / 2;
                const int we = start + max_speech_;
                int best = ws;
                for (int i = ws; i < we; ++i) {
                    if (probs[sf + i] < probs[sf + best]) best = i;
                }
                out[sf + best] = 0;
                start = best + 1;
            }
        }
        return out;
    }

    int smooth_;
    float threshold_;
    int min_speech_, max_speech_, min_silence_, extend_pre_, extend_post_;
};

// ------------------------------------------------- Silero (utils_vad.get_speech_timestamps)

std::vector<Segment> SileroSegments(const ProbTrack& track, float duration_s, const SegmentParams& p) {
    const int sr = kTargetSampleRate;
    const int window = static_cast<int>(std::lround(track.hop_s * sr));
    const int64_t audio_len = static_cast<int64_t>(duration_s * sr);
    const int64_t min_speech = static_cast<int64_t>(sr) * p.min_speech_ms / 1000;
    const int64_t min_silence = static_cast<int64_t>(sr) * p.min_silence_ms / 1000;
    const int64_t pre = static_cast<int64_t>(sr) * p.pre_pad_ms / 1000;
    const int64_t post = static_cast<int64_t>(sr) * p.post_pad_ms / 1000;
    const float neg_threshold = std::max(p.threshold - 0.15f, 0.01f);

    struct S { int64_t start, end; };
    std::vector<S> speeches;
    bool triggered = false;
    int64_t cur_start = 0, temp_end = 0;
    for (size_t i = 0; i < track.probs.size(); ++i) {
        const float prob = track.probs[i];
        const int64_t pos = static_cast<int64_t>(window) * i;
        if (prob >= p.threshold && temp_end) temp_end = 0;
        if (prob >= p.threshold && !triggered) {
            triggered = true;
            cur_start = pos;
            continue;
        }
        if (prob < neg_threshold && triggered) {
            if (!temp_end) temp_end = pos;
            if (pos - temp_end < min_silence) continue;
            if (temp_end - cur_start > min_speech) speeches.push_back({cur_start, temp_end});
            temp_end = 0;
            triggered = false;
        }
    }
    if (triggered && audio_len - cur_start > min_speech) speeches.push_back({cur_start, audio_len});

    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0) speeches[i].start = std::max<int64_t>(0, speeches[i].start - pre);
        if (i + 1 < speeches.size()) {
            const int64_t silence = speeches[i + 1].start - speeches[i].end;
            if (silence < pre + post) {
                // Not enough room for both paddings: share the gap in proportion to them
                // (an even split when pre == post, as in the reference).
                const int64_t to_end = pre == post ? silence / 2 : silence * post / (pre + post);
                const int64_t to_next = pre == post ? silence / 2 : silence - to_end;
                speeches[i].end += to_end;
                speeches[i + 1].start = std::max<int64_t>(0, speeches[i + 1].start - to_next);
            } else {
                speeches[i].end = std::min(audio_len, speeches[i].end + post);
                speeches[i + 1].start = std::max<int64_t>(0, speeches[i + 1].start - pre);
            }
        } else {
            speeches[i].end = std::min(audio_len, speeches[i].end + post);
        }
    }
    std::vector<Segment> segs;
    for (const auto& s : speeches) {
        segs.emplace_back(static_cast<float>(s.start) / sr, static_cast<float>(s.end) / sr);
    }
    return segs;
}

// ------------------------------------------------- YAMNet (simple threshold + merge)

std::vector<Segment> YamnetSegments(const ProbTrack& track, float duration_s, const SegmentParams& p) {
    // The track already holds one probability per hop-sized slot (see YamnetModel::ToSlots):
    // probs[k] covers [k * hop, (k + 1) * hop).
    const float hop = track.hop_s;
    const int slots = static_cast<int>(track.probs.size());
    std::vector<Segment> raw;
    for (int k = 0; k < slots; ++k) {
        if (track.probs[k] < p.threshold) continue;
        const float s = k * hop;
        const float e = std::min(duration_s, (k + 1) * hop);
        if (!raw.empty() && s <= raw.back().second + 1e-4f) {
            raw.back().second = e;
        } else {
            raw.emplace_back(s, e);
        }
    }
    std::vector<Segment> merged;
    for (const auto& seg : raw) {
        if (!merged.empty() && (seg.first - merged.back().second) * 1000 < p.min_silence_ms) {
            merged.back().second = seg.second;
        } else {
            merged.push_back(seg);
        }
    }
    std::vector<Segment> out;
    const float pre = p.pre_pad_ms / 1000.0f;
    const float post = p.post_pad_ms / 1000.0f;
    for (auto seg : merged) {
        if ((seg.second - seg.first) * 1000 < p.min_speech_ms) continue;
        seg.first = std::max(0.0f, seg.first - pre);
        seg.second = std::min(duration_s, seg.second + post);
        if (!out.empty() && seg.first <= out.back().second) {
            out.back().second = seg.second;
        } else {
            out.push_back(seg);
        }
    }
    return out;
}

}  // namespace

std::vector<Segment> MakeSegments(ModelType type, const ProbTrack& track,
                                  float duration_s, const SegmentParams& params) {
    std::vector<Segment> segs;
    switch (type) {
        case ModelType::kFireRedVad: {
            FireRedPostprocessor pp(/*smooth_window=*/5, params.threshold,
                                    std::max(1, params.min_speech_ms / 10),
                                    std::max(1, static_cast<int>(params.max_speech_s * 100)),
                                    params.min_silence_ms / 10, params.pre_pad_ms / 10,
                                    params.post_pad_ms / 10);
            segs = pp.ToSegments(pp.Process(track.probs), duration_s);
            break;
        }
        case ModelType::kSilero:
            segs = SileroSegments(track, duration_s, params);
            break;
        case ModelType::kYamnet:
            segs = YamnetSegments(track, duration_s, params);
            break;
    }
    for (auto& s : segs) {
        s.first = std::round(s.first * 1000) / 1000;
        s.second = std::round(s.second * 1000) / 1000;
    }
    return segs;
}

}  // namespace vad
