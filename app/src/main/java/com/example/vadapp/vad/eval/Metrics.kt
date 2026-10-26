package com.example.vadapp.vad.eval

import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.NativeVad
import com.example.vadapp.vad.Segment
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.VadModelType
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Threshold grid: THRESHOLDS[k] = k / 100. */
const val NUM_THR = 101
fun thresholdAt(k: Int) = k / 100f

enum class EvalMode(val label: String, val short: String) {
    STRICT("Strict (mọi bin)", "all"),
    COLLAR("Collar ±1 bin (bỏ bin biên)", "nb"),
    BOUNDARY("Chỉ bin biên", "b"),
}

/**
 * Cumulative post-processing configurations. A4 = what the user gets with their settings,
 * A5 = the full pipeline with merge/drop/pad found by [ParamSearch] on dev.
 */
enum class PostConfig(val code: String, val description: String) {
    A0("A0", "Điểm bin thô, không hậu xử lý"),
    A1("A1", "Rescore + rebin"),
    A2("A2", "A1 + merge (lặng tối thiểu)"),
    A3("A3", "A2 + drop (speech tối thiểu)"),
    A4("A4", "A3 + pad (cài đặt user)"),
    A5("A5", "Tối ưu: merge/drop/pad tìm trên dev");

    /** Segmenter settings for A1..A5 (A0 does not use the segmenter). */
    fun settings(base: SegmentSettings, tuned: SegmentSettings, threshold: Float) =
        if (this == A5) tuned.copy(threshold = threshold) else base.copy(
            threshold = threshold,
            minSilenceMs = if (this >= A2) base.minSilenceMs else 0,
            minSpeechMs = if (this >= A3) base.minSpeechMs else 0,
            prePadMs = if (this >= A4) base.prePadMs else 0,
            postPadMs = if (this >= A4) base.postPadMs else 0,
        )
}

enum class Criterion(val label: String, val excel: String) {
    DCF_MIN("DCF-min", "DCF-min"),
    F1_MAX("F1-max", "F1-max"),
    YOUDEN("Youden J", "Youden"),
    EER("EER (MR = FAR)", "EER"),
    MR_MAX("MR ≤ x, FAR nhỏ nhất", "MR<=x"),
}

private val NUM_CONFIG = PostConfig.entries.size
private val NUM_MODE = EvalMode.entries.size

/** Flat confusion-count array: [config][threshold][mode][TP, FN, FP, TN]. */
val SWEEP_SIZE = NUM_CONFIG * NUM_THR * NUM_MODE * 4
fun sweepIndex(c: Int, k: Int, mode: Int) = ((c * NUM_THR + k) * NUM_MODE + mode) * 4

data class Conf(val tp: Long, val fn: Long, val fp: Long, val tn: Long) {
    operator fun plus(o: Conf) = Conf(tp + o.tp, fn + o.fn, fp + o.fp, tn + o.tn)
    val mr get() = div(fn, tp + fn)
    val far get() = div(fp, fp + tn)
    val precision get() = div(tp, tp + fp)
    val recall get() = div(tp, tp + fn)
    val f1 get() = div(2 * tp, 2 * tp + fn + fp)
    val specificity get() = div(tn, tn + fp)
    fun dcf(missWeight: Double) = missWeight * mr + (1 - missWeight) * far

    companion object {
        val ZERO = Conf(0, 0, 0, 0)
        private fun div(a: Long, b: Long) = if (b == 0L) Double.NaN else a.toDouble() / b
        fun of(counts: IntArray, at: Int) =
            Conf(counts[at].toLong(), counts[at + 1].toLong(), counts[at + 2].toLong(), counts[at + 3].toLong())
    }
}

/** Everything kept per (video, model) after a sweep; the probabilities themselves are dropped. */
class VideoEval(
    val counts: IntArray,
    /** A0 bin scores and the GT of the evaluated bins (label -1 = skipped), for the exact AUC. */
    val a0Scores: FloatArray,
    val labels: ByteArray,
    val boundary: BooleanArray,
) {
    fun conf(c: Int, k: Int, mode: Int) = Conf.of(counts, sweepIndex(c, k, mode))
}

/**
 * The evaluated 0.5 s bins of one file: GT labels (-1 = skipped), boundary flags and bin
 * lengths (the last bin may be shorter). Holds the one counting rule used everywhere.
 */
class BinGrid(val labels: ByteArray, val boundary: BooleanArray, val binLen: FloatArray) {
    val n: Int get() = labels.size

    /** Adds bin [i] predicted as [speech] at [offsets] = sweep offsets of (strict, collar, boundary). */
    fun add(counts: IntArray, offsets: IntArray, i: Int, speech: Boolean) {
        val q = if (labels[i].toInt() == 1) (if (speech) 0 else 1) else (if (speech) 2 else 3)
        counts[offsets[0] + q]++
        counts[offsets[if (boundary[i]) 2 else 1] + q]++
    }

    /** A bin is speech iff [segs] cover at least [ratio] of it. [frac] is scratch space of size n. */
    fun addSegments(counts: IntArray, offsets: IntArray, segs: List<Segment>, ratio: Float, frac: FloatArray) {
        frac.fill(0f)
        for (s in segs) {
            val b0 = max(0, floor(s.startSec / BIN_SEC).toInt())
            val b1 = min(n - 1, floor(s.endSec / BIN_SEC).toInt())
            for (b in b0..b1) {
                val ov = min(s.endSec, (b + 1) * BIN_SEC) - max(s.startSec, b * BIN_SEC)
                if (ov > 0) frac[b] += ov
            }
        }
        for (i in 0 until n) if (labels[i] >= 0) add(counts, offsets, i, frac[i] / binLen[i] >= ratio - 1e-4f)
    }

    companion object {
        fun of(result: AnalysisResult, gt: GroundTruth): BinGrid {
            val duration = if (result.durationSec > 0) result.durationSec else gt.durationSec
            val n = min(gt.bins, ceil(duration / BIN_SEC - 1e-4f).toInt()).coerceAtLeast(0)
            val labels = gt.labels.copyOf(n)
            val boundary = BooleanArray(n) { i ->
                val l = labels[i]
                l >= 0 && ((i > 0 && labels[i - 1] >= 0 && labels[i - 1] != l) ||
                    (i + 1 < n && labels[i + 1] >= 0 && labels[i + 1] != l))
            }
            val binLen = FloatArray(n) { i -> min(BIN_SEC, duration - i * BIN_SEC).coerceAtLeast(1e-3f) }
            return BinGrid(labels, boundary, binLen)
        }
    }
}

object Metrics {

    /**
     * Sweeps all thresholds for all configs on one file. [ratio] = minimum speech fraction for a
     * 0.5 s bin to count as speech.
     */
    fun evaluateVideo(
        model: VadModelType, result: AnalysisResult, gt: GroundTruth,
        base: SegmentSettings, tuned: SegmentSettings, ratio: Float, keepGoing: () -> Boolean = { true },
    ): VideoEval = evaluate(result, gt, ratio, keepGoing) { cfg, t ->
        NativeVad.segments(model, result, cfg.settings(base, tuned, t))
    }

    /** [segmenter] produces the speech segments of config A1..A5 at threshold t. */
    fun evaluate(
        result: AnalysisResult, gt: GroundTruth, ratio: Float, keepGoing: () -> Boolean = { true },
        segmenter: (PostConfig, Float) -> List<Segment>,
    ): VideoEval {
        val grid = BinGrid.of(result, gt)
        val n = grid.n
        val labels = grid.labels
        val counts = IntArray(SWEEP_SIZE)
        fun offsets(c: Int, k: Int) = IntArray(NUM_MODE) { sweepIndex(c, k, it) }

        // A0: a bin is speech iff at least `ratio` of its frames have p >= t, i.e. iff the
        // ceil(ratio·m)-th largest probability of the bin is >= t. That value is the bin score.
        val perBin = Array(n) { ArrayList<Float>() }
        for (j in result.probs.indices) {
            val b = floor((result.offsetSec + j * result.hopSec) / BIN_SEC).toInt()
            if (b in 0 until n) perBin[b] += result.probs[j]
        }
        val scores = FloatArray(n) { i ->
            val p = perBin[i]
            if (p.isEmpty()) 0f else {
                p.sortDescending()
                p[(ceil(ratio * p.size - 1e-6).toInt()).coerceIn(1, p.size) - 1]
            }
        }
        for (k in 0 until NUM_THR) {
            val t = thresholdAt(k)
            val off = offsets(0, k)
            for (i in 0 until n) if (labels[i] >= 0) grid.add(counts, off, i, scores[i] >= t)
        }

        // A1..A5: run the model's own segmenter, then a bin is speech iff the segments cover
        // at least `ratio` of it.
        val frac = FloatArray(n)
        for (cfg in PostConfig.entries.drop(1)) {
            for (k in 0 until NUM_THR) {
                if (!keepGoing()) throw kotlinx.coroutines.CancellationException()
                grid.addSegments(counts, offsets(cfg.ordinal, k), segmenter(cfg, thresholdAt(k)), ratio, frac)
            }
        }
        return VideoEval(counts, scores, labels, grid.boundary)
    }

    fun sum(videos: List<VideoEval>): IntArray {
        val out = IntArray(SWEEP_SIZE)
        for (v in videos) for (i in out.indices) out[i] += v.counts[i]
        return out
    }

    fun conf(sweep: IntArray, c: Int, k: Int, mode: Int) = Conf.of(sweep, sweepIndex(c, k, mode))

    /** ROC points (FAR, TPR) over the threshold grid. */
    fun roc(sweep: IntArray, c: Int, mode: Int): List<Pair<Double, Double>> =
        (0 until NUM_THR).map { k -> conf(sweep, c, k, mode) }
            .filter { !it.far.isNaN() && !it.recall.isNaN() }
            .map { it.far to it.recall }

    /** Trapezoidal AUC of the grid ROC, closed with (0,0) and (1,1). */
    fun auc(points: List<Pair<Double, Double>>): Double {
        if (points.isEmpty()) return Double.NaN
        val pts = (points + listOf(0.0 to 0.0, 1.0 to 1.0)).sortedWith(compareBy({ it.first }, { it.second }))
        var a = 0.0
        for (i in 1 until pts.size) a += (pts[i].first - pts[i - 1].first) * (pts[i].second + pts[i - 1].second) / 2
        return a
    }

    /** Threshold-free AUC of the A0 bin scores (Mann–Whitney, ties count 1/2). */
    fun exactAuc(videos: List<VideoEval>, mode: Int): Double {
        val pos = ArrayList<Float>()
        val neg = ArrayList<Float>()
        for (v in videos) for (i in v.labels.indices) {
            val l = v.labels[i].toInt()
            if (l < 0) continue
            val inMode = when (mode) { 0 -> true; 1 -> !v.boundary[i]; else -> v.boundary[i] }
            if (!inMode) continue
            if (l == 1) pos += v.a0Scores[i] else neg += v.a0Scores[i]
        }
        if (pos.isEmpty() || neg.isEmpty()) return Double.NaN
        val all = (pos.map { it to true } + neg.map { it to false }).sortedBy { it.first }
        var rankSumPos = 0.0
        var i = 0
        while (i < all.size) {
            var j = i
            while (j + 1 < all.size && all[j + 1].first == all[i].first) j++
            val avgRank = (i + j) / 2.0 + 1
            for (t in i..j) if (all[t].second) rankSumPos += avgRank
            i = j + 1
        }
        val np = pos.size.toDouble()
        return (rankSumPos - np * (np + 1) / 2) / (np * neg.size)
    }

    /** Highest TPR with FAR <= [maxFar] (0 when none), as the Excel formula computes it. */
    fun tprAtFar(sweep: IntArray, c: Int, mode: Int, maxFar: Double): Double =
        (0 until NUM_THR).map { conf(sweep, c, it, mode) }
            .filter { !it.far.isNaN() && !it.recall.isNaN() && it.far <= maxFar + 1e-12 }
            .maxOfOrNull { it.recall } ?: 0.0

    /**
     * Index of the best threshold by [criterion], or -1. Ties go to the lowest threshold
     * (same as MATCH in the exported workbook).
     */
    fun select(sweep: IntArray, c: Int, mode: Int, criterion: Criterion, missWeight: Double, mrMax: Double): Int =
        selectBy(NUM_THR, { conf(sweep, c, it, mode) }, criterion, missWeight, mrMax).first

    /** Best index among [n] candidates and its [cost]; (-1, NaN) if none is eligible. First wins ties. */
    fun selectBy(
        n: Int, confAt: (Int) -> Conf, criterion: Criterion, missWeight: Double, mrMax: Double,
    ): Pair<Int, Double> {
        var best = -1
        var bestV = Double.NaN
        for (k in 0 until n) {
            val v = cost(confAt(k), criterion, missWeight, mrMax)
            if (v.isNaN()) continue
            if (best < 0 || v < bestV) { best = k; bestV = v }
        }
        return best to bestV
    }

    /** Value to minimise for [criterion]; NaN = not eligible. */
    fun cost(x: Conf, criterion: Criterion, missWeight: Double, mrMax: Double): Double = when (criterion) {
        Criterion.DCF_MIN -> x.dcf(missWeight)
        Criterion.F1_MAX -> -x.f1
        Criterion.YOUDEN -> -(x.recall - x.far)
        Criterion.EER -> abs(x.mr - x.far)
        Criterion.MR_MAX -> if (!x.mr.isNaN() && x.mr <= mrMax + 1e-12) x.far else Double.NaN
    }

    /** [cost] back in the criterion's own units (F1 and J are maximised). */
    fun costToValue(criterion: Criterion, cost: Double) =
        if (criterion == Criterion.F1_MAX || criterion == Criterion.YOUDEN) -cost else cost
}

/** Percentile bootstrap over files (resampled with replacement). */
object Bootstrap {
    fun ci(
        perVideo: List<Conf>, metric: (Conf) -> Double,
        iterations: Int = 1000, seed: Long = 42,
    ): Pair<Double, Double>? {
        if (perVideo.isEmpty()) return null
        val rnd = java.util.Random(seed)
        val values = ArrayList<Double>(iterations)
        repeat(iterations) {
            var tp = 0L; var fn = 0L; var fp = 0L; var tn = 0L
            repeat(perVideo.size) {
                val c = perVideo[rnd.nextInt(perVideo.size)]
                tp += c.tp; fn += c.fn; fp += c.fp; tn += c.tn
            }
            val v = metric(Conf(tp, fn, fp, tn))
            if (!v.isNaN()) values += v
        }
        if (values.size < iterations / 2) return null
        values.sort()
        fun q(p: Double) = values[((values.size - 1) * p).toInt().coerceIn(0, values.size - 1)]
        return q(0.025) to q(0.975)
    }
}
