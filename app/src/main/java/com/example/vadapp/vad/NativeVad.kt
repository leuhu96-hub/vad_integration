package com.example.vadapp.vad

import android.content.res.AssetManager

fun interface ProgressListener {
    /** [fraction] in 0..1. Return false to cancel the analysis. */
    fun onProgress(fraction: Float): Boolean
}

/**
 * Output of a model run, before segmentation. Segmentation is a separate, cheap native call
 * so thresholds can be tuned without decoding the video again.
 *
 * probs[i] is the speech probability of the audio centred at offsetSec + i * hopSec.
 */
class AnalysisResult(
    val probs: FloatArray,
    val hopSec: Float,
    val offsetSec: Float,
    val durationSec: Float,
    val sourceSampleRate: Int,
    val sourceChannels: Int,
    val mime: String,
    val elapsedMs: Long,
)

data class Segment(val startSec: Float, val endSec: Float) {
    val durationSec get() = endSec - startSec
    val startMs get() = (startSec * 1000).toLong()
    val endMs get() = (endSec * 1000).toLong()
}

object NativeVad {
    init {
        System.loadLibrary("vadnative")
    }

    /**
     * Decodes the first audio track behind [fd] (any container/codec/sample rate/channel
     * count the device supports), converts it to 16 kHz mono and runs [modelType] on it.
     * Throws RuntimeException on failure and CancellationException when [listener] returns false.
     */
    external fun analyze(
        assets: AssetManager,
        modelType: Int,
        fd: Int,
        offset: Long,
        length: Long,
        numThreads: Int,
        listener: ProgressListener,
    ): AnalysisResult

    /** Returns flattened [start0, end0, start1, end1, ...] in seconds. */
    external fun segment(
        modelType: Int,
        probs: FloatArray,
        hopSec: Float,
        offsetSec: Float,
        durationSec: Float,
        threshold: Float,
        minSpeechMs: Int,
        minSilenceMs: Int,
        padMs: Int,
    ): FloatArray

    fun segments(model: VadModelType, result: AnalysisResult, settings: SegmentSettings): List<Segment> {
        val flat = segment(
            model.id, result.probs, result.hopSec, result.offsetSec, result.durationSec,
            settings.threshold, settings.minSpeechMs, settings.minSilenceMs, settings.padMs,
        )
        return List(flat.size / 2) { Segment(flat[2 * it], flat[2 * it + 1]) }
    }
}
