#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace vad {

struct AudioInfo {
    std::string mime;
    int sample_rate = 0;   // source rate as reported by the decoder
    int channels = 0;      // source channel count as reported by the decoder
    double duration_s = 0;  // container duration (may be 0 if unknown)
};

// Receives 16 kHz mono float samples in [-1, 1].
using PcmSink = std::function<void(const float* samples, size_t n)>;
// Receives decode progress in [0, 1]; return false to cancel.
using ProgressFn = std::function<bool(float)>;

// Decodes the first audio track of any container Android's NDK media stack understands
// (mp4, mkv, webm, 3gp, ts, mp3, wav, ...), downmixes every channel layout to mono and
// resamples any rate to 16 kHz, streaming the result into `sink`.
// Throws VadError on failure and Cancelled when `progress` returns false.
AudioInfo DecodeToMono16k(int fd, int64_t offset, int64_t length,
                          const PcmSink& sink, const ProgressFn& progress);

}  // namespace vad
