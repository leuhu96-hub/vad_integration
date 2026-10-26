package com.example.vadapp.vad.eval

import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.Segment
import com.example.vadapp.vad.SegmentSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

/** Candidate values of the post-processing parameters (ms) and the threshold step (in 1/100). */
data class SearchGrid(
    val merge: List<Int> = listOf(0, 100, 200, 300, 500, 800),
    val drop: List<Int> = listOf(0, 100, 200, 300, 500),
    val prePad: List<Int> = listOf(0, 50, 100, 200),
    val postPad: List<Int> = listOf(0, 50, 100, 200),
    val thrStep: Int = 5,
    /** FireRedVAD smoothing window (ms); only searched when the model has one (see [ParamSearch.search]). */
    val smooth: List<Int> = listOf(30, 50, 100, 150),
) {
    /** Threshold indexes (k / 100) tried during the search. */
    val thresholds: List<Int> get() = (0..100 step thrStep.coerceIn(1, 100)).toList()

    /**
     * All combinations, smoothing (when [withSmooth]) then merge outermost; threshold is filled in
     * later. Parameters the grid does not search come from [base].
     */
    fun combos(base: SegmentSettings = ZERO, withSmooth: Boolean = false): List<SegmentSettings> =
        (if (withSmooth) smooth else listOf(base.smoothWindowMs)).flatMap { sm ->
            merge.flatMap { m ->
                drop.flatMap { d ->
                    prePad.flatMap { pre ->
                        postPad.map { post ->
                            base.copy(
                                threshold = 0f, minSpeechMs = d, minSilenceMs = m, prePadMs = pre, postPadMs = post,
                                smoothWindowMs = sm,
                            )
                        }
                    }
                }
            }
        }

    fun size(withSmooth: Boolean = false): Int =
        merge.size * drop.size * prePad.size * postPad.size * (if (withSmooth) smooth.size else 1)

    companion object {
        /** "0, 100,200" -> [0, 100, 200]; sorted, distinct, non-negative. */
        fun parse(text: String): List<Int> =
            text.split(Regex("[,;\\s]+")).mapNotNull { it.trim().toIntOrNull() }.filter { it >= 0 }.distinct().sorted()

        fun format(values: List<Int>) = values.joinToString(", ")

        private val ZERO = SegmentSettings(threshold = 0f, minSpeechMs = 0, minSilenceMs = 0, prePadMs = 0, postPadMs = 0)
    }
}

/** One tried combination with its best threshold on dev (strict). */
class SearchEntry(
    /** threshold = best threshold of this combination (0 when none was eligible). */
    val settings: SegmentSettings,
    /** Best threshold index (k / 100) or -1. */
    val bestK: Int,
    /** [Metrics.cost] at the best threshold (lower is better), NaN when none is eligible. */
    val cost: Double,
    /** Dev confusion per mode at the best threshold. */
    val conf: List<Conf>,
)

class SearchResult(
    val grid: SearchGrid,
    /** Whether the smoothing window was part of the search (FireRedVAD). */
    val withSmooth: Boolean,
    /** In grid order. */
    val entries: List<SearchEntry>,
    val criterion: Criterion,
    val missWeight: Double,
    val mrMax: Double,
    val devFiles: Int,
    val elapsedMs: Long,
) {
    /** First entry with the lowest cost (ties go to the earlier combination in grid order). */
    val best: SearchEntry? = entries.filter { !it.cost.isNaN() }.minByOrNull { it.cost }

    fun ranked(): List<SearchEntry> = entries.sortedWith(compareBy<SearchEntry> { it.cost.isNaN() }.thenBy { it.cost })

    fun value(e: SearchEntry) = Metrics.costToValue(criterion, e.cost)

    /**
     * Influence of one parameter: for each of its candidate values, the best objective value
     * over all other parameters. [param] 0 = merge, 1 = drop, 2 = pre pad, 3 = post pad,
     * 4 = smoothing (only when [withSmooth]).
     */
    fun marginal(param: Int): List<Pair<Int, Double>> {
        val values = when (param) { 0 -> grid.merge; 1 -> grid.drop; 2 -> grid.prePad; 3 -> grid.postPad; else -> grid.smooth }
        return values.map { v ->
            val best = entries.filter { !it.cost.isNaN() && paramOf(it.settings, param) == v }.minOfOrNull { it.cost }
            v to (best?.let { Metrics.costToValue(criterion, it) } ?: Double.NaN)
        }
    }

    /** Parameter indexes of [marginal] that were searched. */
    val params: IntRange get() = if (withSmooth) 0..4 else 0..3

    companion object {
        val PARAM_NAMES = listOf("Merge (lặng ≥ ms)", "Drop (speech < ms)", "Pre pad (ms)", "Post pad (ms)", "Làm mượt (ms)")

        fun paramOf(s: SegmentSettings, param: Int) = when (param) {
            0 -> s.minSilenceMs; 1 -> s.minSpeechMs; 2 -> s.prePadMs; 3 -> s.postPadMs; else -> s.smoothWindowMs
        }
    }
}

/**
 * Exhaustive grid search of [smoothing ×] merge / drop / pre pad / post pad (× threshold) on the dev files,
 * scored by the selection criterion on the strict mode.
 */
object ParamSearch {

    private val MODES = EvalMode.entries.size

    suspend fun search(
        devFiles: Int,
        /** Model output and GT of dev file i, or null if it has none (skipped). */
        load: (Int) -> Pair<AnalysisResult, GroundTruth>?,
        segmenter: (AnalysisResult, SegmentSettings) -> List<Segment>,
        grid: SearchGrid,
        ratio: Float,
        criterion: Criterion,
        missWeight: Double,
        mrMax: Double,
        keepGoing: () -> Boolean = { true },
        onFileDone: (done: Int) -> Unit = {},
        /** Source of the parameters the grid does not search (e.g. FireRedVAD max speech). */
        base: SegmentSettings? = null,
        /** Also search [SearchGrid.smooth] (only meaningful for FireRedVAD). */
        withSmooth: Boolean = false,
    ): SearchResult {
        val started = System.currentTimeMillis()
        val combos = if (base != null) grid.combos(base, withSmooth) else grid.combos(withSmooth = withSmooth)
        val thr = grid.thresholds
        val t = thr.size
        fun at(c: Int, ti: Int, mode: Int) = ((c * t + ti) * MODES + mode) * 4
        val total = IntArray(combos.size * t * MODES * 4)
        val lock = Any()
        var done = 0

        coroutineScope {
            (0 until devFiles).map { f ->
                async(Dispatchers.Default) {
                    val (result, gt) = load(f) ?: return@async
                    val bins = BinGrid.of(result, gt)
                    val counts = IntArray(total.size)
                    val frac = FloatArray(bins.n)
                    val offsets = IntArray(MODES)
                    combos.forEachIndexed { c, combo ->
                        if (!keepGoing()) throw CancellationException()
                        thr.forEachIndexed { ti, k ->
                            for (m in 0 until MODES) offsets[m] = at(c, ti, m)
                            bins.addSegments(counts, offsets, segmenter(result, combo.copy(threshold = thresholdAt(k))), ratio, frac)
                        }
                    }
                    val n = synchronized(lock) {
                        for (i in total.indices) total[i] += counts[i]
                        ++done
                    }
                    onFileDone(n)
                }
            }.awaitAll()
        }

        val strict = EvalMode.STRICT.ordinal
        val entries = combos.mapIndexed { c, combo ->
            val (ti, cost) = Metrics.selectBy(t, { Conf.of(total, at(c, it, strict)) }, criterion, missWeight, mrMax)
            val k = if (ti >= 0) thr[ti] else -1
            SearchEntry(
                settings = combo.copy(threshold = if (k >= 0) thresholdAt(k) else 0f),
                bestK = k,
                cost = cost,
                conf = List(MODES) { m -> if (ti >= 0) Conf.of(total, at(c, ti, m)) else Conf.ZERO },
            )
        }
        return SearchResult(grid, withSmooth, entries, criterion, missWeight, mrMax, devFiles, System.currentTimeMillis() - started)
    }
}
