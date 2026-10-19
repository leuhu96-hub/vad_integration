#pragma once

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace vad {

// Log-Mel filterbank matching kaldi-native-fbank defaults as used by FireRedVAD:
// 16 kHz, 25 ms window / 10 ms shift, 80 bins, povey window, pre-emphasis 0.97,
// DC removal, power spectrum, low_freq 20 Hz, high_freq Nyquist, snip_edges, no dither.
// Input samples are expected on the int16 scale (i.e. [-32768, 32767]), like Kaldi.
class Fbank {
public:
    static constexpr int kFrameLength = 400;
    static constexpr int kFrameShift = 160;
    static constexpr int kFftSize = 512;
    static constexpr int kNumBins = 80;

    Fbank() {
        window_.resize(kFrameLength);
        const double a = 2.0 * M_PI / (kFrameLength - 1);
        for (int i = 0; i < kFrameLength; ++i) {
            window_[i] = static_cast<float>(std::pow(0.5 - 0.5 * std::cos(a * i), 0.85));
        }
        InitMel();
        InitFft();
    }

    static int NumFrames(size_t num_samples) {
        if (num_samples < static_cast<size_t>(kFrameLength)) return 0;
        return 1 + static_cast<int>((num_samples - kFrameLength) / kFrameShift);
    }

    // Computes `num_frames` frames starting at `wave`; output is [num_frames * kNumBins].
    void Compute(const float* wave, int num_frames, float* out) {
        std::vector<float> frame(kFrameLength);
        std::vector<std::complex<float>> spec(kFftSize);
        std::vector<float> power(kFftSize / 2 + 1);
        for (int f = 0; f < num_frames; ++f) {
            const float* src = wave + static_cast<size_t>(f) * kFrameShift;
            double mean = 0.0;
            for (int i = 0; i < kFrameLength; ++i) mean += src[i];
            mean /= kFrameLength;
            for (int i = 0; i < kFrameLength; ++i) frame[i] = static_cast<float>(src[i] - mean);
            for (int i = kFrameLength - 1; i > 0; --i) frame[i] -= 0.97f * frame[i - 1];
            frame[0] -= 0.97f * frame[0];
            for (int i = 0; i < kFrameLength; ++i) spec[i] = {frame[i] * window_[i], 0.0f};
            for (int i = kFrameLength; i < kFftSize; ++i) spec[i] = {0.0f, 0.0f};
            Fft(spec);
            for (int i = 0; i <= kFftSize / 2; ++i) power[i] = std::norm(spec[i]);

            float* dst = out + static_cast<size_t>(f) * kNumBins;
            for (int b = 0; b < kNumBins; ++b) {
                float e = 0.0f;
                const auto& w = mel_weights_[b];
                for (size_t k = 0; k < w.size(); ++k) e += w[k] * power[mel_first_[b] + k];
                dst[b] = std::log(std::max(e, FLT_EPSILON));
            }
        }
    }

private:
    static float MelScale(float f) { return 1127.0f * std::log(1.0f + f / 700.0f); }

    void InitMel() {
        const int num_fft_bins = kFftSize / 2;
        const float bin_width = 16000.0f / kFftSize;
        const float mel_low = MelScale(20.0f);
        const float mel_high = MelScale(8000.0f);
        const float delta = (mel_high - mel_low) / (kNumBins + 1);
        mel_first_.resize(kNumBins);
        mel_weights_.resize(kNumBins);
        for (int b = 0; b < kNumBins; ++b) {
            const float left = mel_low + b * delta;
            const float center = mel_low + (b + 1) * delta;
            const float right = mel_low + (b + 2) * delta;
            int first = -1;
            std::vector<float> w;
            for (int i = 0; i < num_fft_bins; ++i) {
                const float mel = MelScale(bin_width * i);
                if (mel > left && mel < right) {
                    const float weight = mel <= center ? (mel - left) / (center - left)
                                                       : (right - mel) / (right - center);
                    if (first < 0) first = i;
                    // Keep weights contiguous from `first` (zeros cannot appear in between).
                    w.resize(i - first + 1, 0.0f);
                    w[i - first] = weight;
                }
            }
            mel_first_[b] = first < 0 ? 0 : first;
            mel_weights_[b] = std::move(w);
        }
    }

    void InitFft() {
        int bits = 0;
        while ((1 << bits) < kFftSize) ++bits;
        bitrev_.resize(kFftSize);
        for (int i = 0; i < kFftSize; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
            bitrev_[i] = r;
        }
        twiddle_.resize(kFftSize / 2);
        for (int i = 0; i < kFftSize / 2; ++i) {
            const double a = -2.0 * M_PI * i / kFftSize;
            twiddle_[i] = {static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a))};
        }
    }

    void Fft(std::vector<std::complex<float>>& x) const {
        for (int i = 0; i < kFftSize; ++i) {
            if (i < bitrev_[i]) std::swap(x[i], x[bitrev_[i]]);
        }
        for (int len = 2; len <= kFftSize; len <<= 1) {
            const int half = len / 2;
            const int stride = kFftSize / len;
            for (int i = 0; i < kFftSize; i += len) {
                for (int j = 0; j < half; ++j) {
                    const std::complex<float> t = twiddle_[j * stride] * x[i + j + half];
                    x[i + j + half] = x[i + j] - t;
                    x[i + j] += t;
                }
            }
        }
    }

    std::vector<float> window_;
    std::vector<int> mel_first_;
    std::vector<std::vector<float>> mel_weights_;
    std::vector<int> bitrev_;
    std::vector<std::complex<float>> twiddle_;
};

// Kaldi CMVN statistics (binary "DM"/"FM" matrix, 2 x (dim+1)) -> mean / inverse std.
struct Cmvn {
    std::vector<float> mean;
    std::vector<float> istd;
    void Apply(float* feat, int num_frames) const {
        const int dim = static_cast<int>(mean.size());
        for (int f = 0; f < num_frames; ++f) {
            float* row = feat + static_cast<size_t>(f) * dim;
            for (int d = 0; d < dim; ++d) row[d] = (row[d] - mean[d]) * istd[d];
        }
    }
};

}  // namespace vad
