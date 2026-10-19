#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace vad {

// Streaming band-limited (Kaiser-windowed sinc) resampler for mono float audio.
// Output sample k is taken at input time k * in_rate / out_rate.
class Resampler {
public:
    Resampler(int in_rate, int out_rate, int zero_crossings = 16)
        : in_rate_(in_rate), out_rate_(out_rate),
          step_(static_cast<double>(in_rate) / out_rate) {
        if (in_rate_ == out_rate_) return;
        // Low-pass at the lower Nyquist, with a little headroom for the transition band.
        cutoff_ = 0.97 * std::min(1.0, static_cast<double>(out_rate) / in_rate);
        half_width_ = static_cast<int>(std::ceil(zero_crossings / cutoff_));
        const int n = half_width_ * kTableRes + 2;
        table_.resize(n);
        const double beta = 8.0;
        const double i0b = BesselI0(beta);
        for (int i = 0; i < n; ++i) {
            const double x = static_cast<double>(i) / kTableRes;  // in input samples
            const double r = x / half_width_;
            double w = r >= 1.0 ? 0.0 : BesselI0(beta * std::sqrt(1.0 - r * r)) / i0b;
            const double a = M_PI * cutoff_ * x;
            const double sinc = x == 0.0 ? 1.0 : std::sin(a) / a;
            table_[i] = static_cast<float>(cutoff_ * sinc * w);
        }
        // Left history so the first output sample has full support.
        buf_.assign(half_width_, 0.0f);
        pos_ = half_width_;
    }

    bool passthrough() const { return in_rate_ == out_rate_; }

    void Process(const float* in, size_t n, std::vector<float>& out) {
        in_total_ += n;
        if (passthrough()) {
            out.insert(out.end(), in, in + n);
            out_total_ += n;
            return;
        }
        buf_.insert(buf_.end(), in, in + n);
        Drain(out, OutputsWanted(false));
    }

    void Flush(std::vector<float>& out) {
        if (passthrough()) return;
        buf_.insert(buf_.end(), half_width_ + 1, 0.0f);
        Drain(out, OutputsWanted(true));
    }

private:
    static constexpr int kTableRes = 512;

    static double BesselI0(double x) {
        double sum = 1.0, term = 1.0;
        const double q = x * x / 4.0;
        for (int k = 1; k < 64; ++k) {
            term *= q / (static_cast<double>(k) * k);
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    }

    // Total number of output samples that should exist for the input seen so far.
    size_t OutputsWanted(bool final) const {
        const double exact = static_cast<double>(in_total_) * out_rate_ / in_rate_;
        return static_cast<size_t>(final ? std::ceil(exact) : std::floor(exact));
    }

    // Sum over taps at distances d0, d0 + 1, ... (in input samples, d0 in [0, 1)) walking
    // away from `src` in direction `dir`. Consecutive taps are exactly kTableRes table
    // entries apart, so the interpolation fraction is the same for every tap.
    float Side(const float* src, int dir, double d0, int taps) const {
        const double x = d0 * kTableRes;
        const int i0 = static_cast<int>(x);
        const float f = static_cast<float>(x - i0);
        const float* t = table_.data() + i0;
        float acc = 0.0f;
        for (int k = 0; k < taps; ++k, t += kTableRes, src += dir) {
            acc += *src * (t[0] + f * (t[1] - t[0]));
        }
        return acc;
    }

    void Drain(std::vector<float>& out, size_t wanted) {
        const int buf_size = static_cast<int>(buf_.size());
        while (out_total_ < wanted) {
            const int center = static_cast<int>(pos_);
            if (center + half_width_ >= buf_size) break;
            const double frac = pos_ - center;
            // Taps center, center-1, ... (distance frac + k) and center+1, ... (distance 1 - frac + k).
            const float acc = Side(&buf_[center], -1, frac, half_width_) +
                              Side(&buf_[center + 1], +1, 1.0 - frac, half_width_);
            out.push_back(acc);
            ++out_total_;
            pos_ += step_;
        }
        // Drop samples that no future output can reach.
        const int keep_from = static_cast<int>(pos_) - half_width_ + 1;
        if (keep_from > 0) {
            const int drop = std::min(keep_from, buf_size);
            buf_.erase(buf_.begin(), buf_.begin() + drop);
            pos_ -= drop;
        }
    }

    int in_rate_, out_rate_;
    double step_;
    double cutoff_ = 1.0;
    int half_width_ = 0;
    std::vector<float> table_;
    std::vector<float> buf_;
    double pos_ = 0.0;
    size_t in_total_ = 0;
    size_t out_total_ = 0;
};

}  // namespace vad
