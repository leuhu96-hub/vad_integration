package com.example.vadapp.vad

data class SegmentSettings(
    val threshold: Float,
    val minSpeechMs: Int,
    val minSilenceMs: Int,
    val padMs: Int,
)

/** [id] must match vad::ModelType in native code. Defaults follow each model's reference code. */
enum class VadModelType(val id: Int, val label: String, val description: String, val defaults: SegmentSettings) {
    FIRERED(
        0, "FireRedVAD", "DFSMN, fbank 80 + CMVN, khung 10 ms",
        SegmentSettings(threshold = 0.4f, minSpeechMs = 200, minSilenceMs = 200, padMs = 0),
    ),
    SILERO(
        1, "Silero", "Silero VAD v6, cửa sổ 32 ms",
        SegmentSettings(threshold = 0.5f, minSpeechMs = 250, minSilenceMs = 100, padMs = 30),
    ),
    YAMNET(
        2, "YAMNet", "AudioSet 521 lớp, lớp Speech, bước 0.48 s",
        SegmentSettings(threshold = 0.3f, minSpeechMs = 250, minSilenceMs = 500, padMs = 0),
    ),
}
