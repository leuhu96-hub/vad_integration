#include "audio_decoder.h"

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "common.h"
#include "resampler.h"

namespace vad {
namespace {

// android.media.AudioFormat encodings.
constexpr int kPcm16 = 2;
constexpr int kPcm8 = 3;
constexpr int kPcmFloat = 4;
constexpr int kPcm24Packed = 21;
constexpr int kPcm32 = 22;

constexpr int64_t kTimeoutUs = 10000;
constexpr int64_t kFeedTimeoutUs = 1000;

struct ExtractorDeleter {
    void operator()(AMediaExtractor* e) const { AMediaExtractor_delete(e); }
};
struct FormatDeleter {
    void operator()(AMediaFormat* f) const { AMediaFormat_delete(f); }
};
struct CodecDeleter {
    void operator()(AMediaCodec* c) const {
        AMediaCodec_stop(c);
        AMediaCodec_delete(c);
    }
};

int BytesPerSample(int encoding) {
    switch (encoding) {
        case kPcm8: return 1;
        case kPcm16: return 2;
        case kPcm24Packed: return 3;
        case kPcmFloat:
        case kPcm32: return 4;
        default: throw VadError("Unsupported PCM encoding " + std::to_string(encoding));
    }
}

float SampleAt(const uint8_t* p, int encoding) {
    switch (encoding) {
        case kPcm8: return (static_cast<int>(p[0]) - 128) / 128.0f;
        case kPcm16: {
            int16_t v;
            std::memcpy(&v, p, 2);
            return v / 32768.0f;
        }
        case kPcm24Packed: {
            int32_t v = (p[0] << 8) | (p[1] << 16) | (p[2] << 24);
            return (v >> 8) / 8388608.0f;
        }
        case kPcmFloat: {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        case kPcm32: {
            int32_t v;
            std::memcpy(&v, p, 4);
            return v / 2147483648.0f;
        }
    }
    return 0.0f;
}

// Converts interleaved PCM of any layout to mono, then resamples to 16 kHz.
class MonoConverter {
public:
    explicit MonoConverter(const PcmSink& sink) : sink_(sink) {}

    void SetFormat(int sample_rate, int channels, int encoding) {
        if (sample_rate <= 0 || channels <= 0) throw VadError("Invalid audio output format");
        if (resampler_ && sample_rate != sample_rate_) Flush();
        if (!resampler_ || sample_rate != sample_rate_) {
            resampler_ = std::make_unique<Resampler>(sample_rate, kTargetSampleRate);
        }
        sample_rate_ = sample_rate;
        channels_ = channels;
        encoding_ = encoding;
        frame_bytes_ = BytesPerSample(encoding) * channels;
    }

    void Push(const uint8_t* data, size_t size) {
        // A codec may hand out a partial frame at a buffer boundary; carry it over.
        carry_.insert(carry_.end(), data, data + size);
        const size_t frames = carry_.size() / frame_bytes_;
        if (frames == 0) return;
        mono_.resize(frames);
        const int bps = BytesPerSample(encoding_);
        const float scale = 1.0f / channels_;
        const uint8_t* p = carry_.data();
        for (size_t i = 0; i < frames; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < channels_; ++c, p += bps) acc += SampleAt(p, encoding_);
            mono_[i] = acc * scale;
        }
        carry_.erase(carry_.begin(), carry_.begin() + frames * frame_bytes_);
        out_.clear();
        resampler_->Process(mono_.data(), frames, out_);
        if (!out_.empty()) sink_(out_.data(), out_.size());
    }

    void Flush() {
        if (!resampler_) return;
        out_.clear();
        resampler_->Flush(out_);
        if (!out_.empty()) sink_(out_.data(), out_.size());
        resampler_.reset();
    }

private:
    const PcmSink& sink_;
    std::unique_ptr<Resampler> resampler_;
    int sample_rate_ = 0, channels_ = 0, encoding_ = kPcm16;
    size_t frame_bytes_ = 2;
    std::vector<uint8_t> carry_;
    std::vector<float> mono_, out_;
};

void ReadOutputFormat(AMediaFormat* fmt, int* sr, int* ch, int* enc) {
    int32_t v;
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_SAMPLE_RATE, &v)) *sr = v;
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &v)) *ch = v;
    // AMEDIAFORMAT_KEY_PCM_ENCODING is only declared from API 28; the key string works everywhere.
    if (AMediaFormat_getInt32(fmt, "pcm-encoding", &v)) *enc = v;
}

}  // namespace

AudioInfo DecodeToMono16k(int fd, int64_t offset, int64_t length,
                          const PcmSink& sink, const ProgressFn& progress) {
    std::unique_ptr<AMediaExtractor, ExtractorDeleter> ex(AMediaExtractor_new());
    if (AMediaExtractor_setDataSourceFd(ex.get(), fd, offset, length) != AMEDIA_OK) {
        throw VadError("Cannot open media file");
    }

    AudioInfo info;
    std::unique_ptr<AMediaFormat, FormatDeleter> track_fmt;
    const size_t num_tracks = AMediaExtractor_getTrackCount(ex.get());
    for (size_t i = 0; i < num_tracks; ++i) {
        std::unique_ptr<AMediaFormat, FormatDeleter> fmt(AMediaExtractor_getTrackFormat(ex.get(), i));
        const char* mime = nullptr;
        if (AMediaFormat_getString(fmt.get(), AMEDIAFORMAT_KEY_MIME, &mime) && mime &&
            std::strncmp(mime, "audio/", 6) == 0) {
            info.mime = mime;
            AMediaExtractor_selectTrack(ex.get(), i);
            track_fmt = std::move(fmt);
            break;
        }
    }
    if (!track_fmt) throw VadError("No audio track in this file");

    int sample_rate = 0, channels = 0, encoding = kPcm16;
    ReadOutputFormat(track_fmt.get(), &sample_rate, &channels, &encoding);
    int64_t duration_us = 0;
    AMediaFormat_getInt64(track_fmt.get(), AMEDIAFORMAT_KEY_DURATION, &duration_us);
    info.duration_s = duration_us / 1e6;

    std::unique_ptr<AMediaCodec, CodecDeleter> codec(
            AMediaCodec_createDecoderByType(info.mime.c_str()));
    if (!codec) throw VadError("No decoder for " + info.mime);
    if (AMediaCodec_configure(codec.get(), track_fmt.get(), nullptr, nullptr, 0) != AMEDIA_OK ||
        AMediaCodec_start(codec.get()) != AMEDIA_OK) {
        throw VadError("Cannot start decoder for " + info.mime);
    }

    MonoConverter converter(sink);
    bool format_known = false;
    bool input_eos = false, output_eos = false;
    float last_progress = -1.0f;

    while (!output_eos) {
        // Queue as much compressed input as the codec accepts right now, without blocking.
        while (!input_eos) {
            const ssize_t in_idx = AMediaCodec_dequeueInputBuffer(codec.get(), 0);
            if (in_idx < 0) break;
            {
                size_t cap = 0;
                uint8_t* buf = AMediaCodec_getInputBuffer(codec.get(), in_idx, &cap);
                const ssize_t n = AMediaExtractor_readSampleData(ex.get(), buf, cap);
                if (n < 0) {
                    AMediaCodec_queueInputBuffer(codec.get(), in_idx, 0, 0, 0,
                                                 AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
                    input_eos = true;
                } else {
                    const int64_t pts = AMediaExtractor_getSampleTime(ex.get());
                    AMediaCodec_queueInputBuffer(codec.get(), in_idx, 0, n, pts, 0);
                    AMediaExtractor_advance(ex.get());
                }
            }
        }

        AMediaCodecBufferInfo bi;
        // Only wait long once all input is queued; otherwise go back to feeding quickly.
        const ssize_t out_idx = AMediaCodec_dequeueOutputBuffer(
                codec.get(), &bi, input_eos ? kTimeoutUs : kFeedTimeoutUs);
        if (out_idx >= 0) {
            if (bi.size > 0) {
                if (!format_known) {
                    // Some decoders never emit FORMAT_CHANGED; fall back to the track format.
                    std::unique_ptr<AMediaFormat, FormatDeleter> f(AMediaCodec_getOutputFormat(codec.get()));
                    ReadOutputFormat(f.get(), &sample_rate, &channels, &encoding);
                    converter.SetFormat(sample_rate, channels, encoding);
                    format_known = true;
                }
                size_t cap = 0;
                const uint8_t* buf = AMediaCodec_getOutputBuffer(codec.get(), out_idx, &cap);
                converter.Push(buf + bi.offset, bi.size);
            }
            AMediaCodec_releaseOutputBuffer(codec.get(), out_idx, false);
            if (bi.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) output_eos = true;

            if (duration_us > 0 && bi.presentationTimeUs > 0) {
                const float p = std::min(1.0f, static_cast<float>(bi.presentationTimeUs) / duration_us);
                if (p - last_progress >= 0.005f) {
                    last_progress = p;
                    if (!progress(p)) throw Cancelled();
                }
            }
        } else if (out_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            std::unique_ptr<AMediaFormat, FormatDeleter> f(AMediaCodec_getOutputFormat(codec.get()));
            ReadOutputFormat(f.get(), &sample_rate, &channels, &encoding);
            converter.SetFormat(sample_rate, channels, encoding);
            format_known = true;
        } else if (out_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER ||
                   out_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
            // keep pumping
        } else {
            throw VadError("Decoder error " + std::to_string(out_idx));
        }
    }
    converter.Flush();

    info.sample_rate = sample_rate;
    info.channels = channels;
    LOGI("decoded %s: %d Hz, %d ch, pcm-encoding %d, %.2fs", info.mime.c_str(),
         sample_rate, channels, encoding, info.duration_s);
    return info;
}

}  // namespace vad
