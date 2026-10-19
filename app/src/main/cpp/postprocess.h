#pragma once

#include <utility>
#include <vector>

#include "common.h"
#include "vad_models.h"

namespace vad {

struct SegmentParams {
    float threshold = 0.5f;
    int min_speech_ms = 250;
    int min_silence_ms = 100;
    int pad_ms = 0;
    float max_speech_s = 20.0f;  // FireRedVAD splits longer segments at a probability dip
};

using Segment = std::pair<float, float>;  // [start, end) in seconds

// Turns a probability track into speech segments using each model's reference
// post-processing (FireRedVAD VadPostprocessor, Silero get_speech_timestamps).
std::vector<Segment> MakeSegments(ModelType type, const ProbTrack& track,
                                  float duration_s, const SegmentParams& params);

}  // namespace vad
