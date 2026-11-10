package com.example.vadapp.vad

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.net.Uri

fun interface ProgressListener {
    /** [fraction] in 0..1. Return false to cancel the analysis. */
    fun onProgress(fraction: Float): Boolean
}

/**
 * Output of a model run, before segmentation. Segmentation is a separate, cheap native call
 * so thresholds can be tuned without decoding the video again.
 *
 * probs[i] is the speech probability of the audio centred at offsetSec + i * hopSec.
 * For YAMNet these are per-slot values (triangle-weighted over the overlapping windows),
 * the same values the segmenter thresholds.
 */
class AnalysisResult(
    val probs: FloatArray,
    val hopSec: Float,
    val offsetSec: Float,
    val durationSec: Float,
    val sourceSampleRate: Int,
    val sourceChannels: Int,
    val mime: String,
    /** Runtime that executed the model: "ONNX" or "TFLite". */
    val format: String,
    val elapsedMs: Long,
    /** Peak amplitude of the 16 kHz mono signal per [envelopeHopSec] bucket, normalised to 0..1. */
    val envelope: FloatArray,
    val envelopeHopSec: Float,
    /**
     * AED only: the channels after speech (see [AedEvent]) one after another, each [probs].size
     * long, on the same time axis as [probs]. Empty for every other model.
     */
    val eventProbs: FloatArray = FloatArray(0),
) {
    val eventCount: Int get() = if (probs.isEmpty()) 0 else eventProbs.size / probs.size

    fun eventTrack(index: Int): FloatArray = eventProbs.copyOfRange(index * probs.size, (index + 1) * probs.size)
}

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
     *
     * [modelPath] / [extraPath] are "asset:<path>" or absolute file paths. The runtime
     * (ONNX Runtime or TFLite) is picked from the model file's content, not its name.
     * Throws RuntimeException on failure and CancellationException when [listener] returns false.
     */
    external fun analyze(
        assets: AssetManager,
        modelType: Int,
        modelPath: String,
        extraPath: String?,
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
        prePadMs: Int,
        postPadMs: Int,
        smoothWindowMs: Int,
        maxSpeechSec: Float,
    ): FloatArray

    /** Runs [model] with the file [source] on the media behind a content [uri]. */
    fun analyzeUri(
        context: Context, uri: Uri, model: VadModelType, source: ModelSource, listener: ProgressListener,
    ): AnalysisResult {
        val afd = context.contentResolver.openAssetFileDescriptor(uri, "r")
            ?: error("Không mở được file")
        return afd.use {
            val length = if (it.length == AssetFileDescriptor.UNKNOWN_LENGTH) Long.MAX_VALUE else it.length
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            analyze(
                context.assets, model.id, source.path, model.extraAsset,
                it.parcelFileDescriptor.fd, it.startOffset, length, threads, listener,
            )
        }
    }

    fun segments(model: VadModelType, result: AnalysisResult, settings: SegmentSettings): List<Segment> {
        val flat = segment(
            model.id, result.probs, result.hopSec, result.offsetSec, result.durationSec,
            settings.threshold, settings.minSpeechMs, settings.minSilenceMs,
            settings.prePadMs, settings.postPadMs, settings.smoothWindowMs, settings.maxSpeechSec,
        )
        return List(flat.size / 2) { Segment(flat[2 * it], flat[2 * it + 1]) }
    }

    /**
     * Segments of every [AedEvent] channel of an AED [result], thresholded with [thresholds]
     * (one per event, in [AedEvent] order) and cleaned up like speech (same smoothing, minimum
     * length and silence merge as [settings]). Empty map for models without events.
     */
    fun eventSegments(
        result: AnalysisResult, settings: SegmentSettings, thresholds: Map<AedEvent, Float>,
    ): Map<AedEvent, List<Segment>> {
        val n = minOf(result.eventCount, AedEvent.entries.size)
        if (n == 0) return emptyMap()
        return AedEvent.entries.take(n).associateWith { ev ->
            val flat = segment(
                VadModelType.FIRERED.id, result.eventTrack(ev.ordinal), result.hopSec, result.offsetSec,
                result.durationSec, thresholds[ev] ?: ev.defaultThreshold,
                settings.minSpeechMs, settings.minSilenceMs, settings.prePadMs, settings.postPadMs,
                settings.smoothWindowMs, settings.maxSpeechSec,
            )
            List(flat.size / 2) { Segment(flat[2 * it], flat[2 * it + 1]) }
        }
    }
}
