package com.example.vadapp.vad

data class SegmentSettings(
    val threshold: Float,
    val minSpeechMs: Int,
    val minSilenceMs: Int,
    val padMs: Int,
)

/**
 * [id] must match vad::ModelType in native code. Defaults follow each model's reference code.
 * Bundled model files are looked up as `assets/models/<assetBase>.<onnx|tflite>`.
 */
enum class VadModelType(
    val id: Int,
    val label: String,
    val description: String,
    val assetBase: String,
    val defaults: SegmentSettings,
) {
    FIRERED(
        0, "FireRedVAD", "DFSMN, fbank 80 + CMVN, khung 10 ms", "fireredvad_vad",
        SegmentSettings(threshold = 0.4f, minSpeechMs = 200, minSilenceMs = 200, padMs = 0),
    ),
    SILERO(
        1, "Silero", "Silero VAD v6, cửa sổ 32 ms", "silero_vad",
        SegmentSettings(threshold = 0.5f, minSpeechMs = 250, minSilenceMs = 100, padMs = 30),
    ),
    YAMNET(
        2, "YAMNet", "AudioSet 521 lớp, lớp Speech, bước 0.48 s", "yamnet",
        SegmentSettings(threshold = 0.3f, minSpeechMs = 250, minSilenceMs = 500, padMs = 0),
    );

    /** Extra file the native runner needs besides the network itself. */
    val extraAsset: String? get() = if (this == FIRERED) "asset:models/cmvn.ark" else null
}
