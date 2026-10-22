package com.example.vadapp.vad.eval

import kotlin.math.abs

/** isDev[i] tells whether items[i] is in the dev set (otherwise test). */
class Split(val devRatio: Float, val isDev: BooleanArray) {
    fun devIndices() = isDev.indices.filter { isDev[it] }
    fun testIndices() = isDev.indices.filter { !isDev[it] }
}

class CellStats(
    val quality: String,
    val category: String,
    val devCount: Int,
    val testCount: Int,
    val devSec: Float,
    val testSec: Float,
    val devSpeech: Float,
    val testSpeech: Float,
)

/**
 * Balanced dev/test split:
 *  1. stratify by (quality × category) so both sets cover every cell,
 *  2. inside a cell, greedily give the longest files to whichever set is furthest below its
 *     duration target,
 *  3. improve with moves/swaps inside each cell while they lower
 *     mean_cells |devShare − r| + |devShare − r| + |speech_dev − speech_test|.
 * Deterministic: depends only on the item list (sorted by name) and the ratio.
 */
object Splitter {

    fun split(items: List<EvalItem>, devRatio: Float): Split {
        val r = devRatio.toDouble()
        val dur = DoubleArray(items.size) { items[it].gt.labelledBins.toDouble() }
        val sp = DoubleArray(items.size) { items[it].gt.speechBins.toDouble() }
        val isDev = BooleanArray(items.size)
        val cells = items.indices.groupBy { items[it].quality to items[it].category }.values
            .map { idx -> idx.sortedBy { items[it].name } }

        for (cell in cells) {
            if (cell.size == 1) { isDev[cell[0]] = true; continue }
            var devD = 0.0
            var testD = 0.0
            for (i in cell.sortedByDescending { dur[it] }) {
                // Deficit relative to the target share of what has been assigned so far.
                val total = devD + testD + dur[i]
                val devGap = r * total - devD
                val testGap = (1 - r) * total - testD
                if (devGap >= testGap) { isDev[i] = true; devD += dur[i] } else testD += dur[i]
            }
            // Never leave a cell (with >= 2 files) without a test file, or without a dev file.
            if (cell.all { isDev[it] }) isDev[cell.minBy { dur[it] }] = false
            if (cell.none { isDev[it] }) isDev[cell.maxBy { dur[it] }] = true
        }

        fun objective(): Double {
            var devD = 0.0; var allD = 0.0; var devS = 0.0; var testS = 0.0; var testD = 0.0
            var cellTerm = 0.0
            for (cell in cells) {
                var cd = 0.0; var ca = 0.0
                for (i in cell) {
                    ca += dur[i]; allD += dur[i]
                    if (isDev[i]) { cd += dur[i]; devD += dur[i]; devS += sp[i] } else { testD += dur[i]; testS += sp[i] }
                }
                if (ca > 0) cellTerm += abs(cd / ca - r)
            }
            val speechTerm = if (devD > 0 && testD > 0) abs(devS / devD - testS / testD) else 1.0
            return cellTerm / cells.size.coerceAtLeast(1) + abs(devD / allD.coerceAtLeast(1e-9) - r) + speechTerm
        }

        var best = objective()
        repeat(50) {
            var improved = false
            for (cell in cells) {
                if (cell.size < 2) continue
                // Swaps keep the per-cell file counts; single moves let the counts adjust.
                for (a in cell) for (b in cell) {
                    if (a >= b || isDev[a] == isDev[b]) continue
                    isDev[a] = !isDev[a]; isDev[b] = !isDev[b]
                    val v = objective()
                    if (v < best - 1e-12) { best = v; improved = true } else { isDev[a] = !isDev[a]; isDev[b] = !isDev[b] }
                }
                for (a in cell) {
                    isDev[a] = !isDev[a]
                    val ok = cell.any { isDev[it] } && cell.any { !isDev[it] }
                    val v = if (ok) objective() else Double.MAX_VALUE
                    if (v < best - 1e-12) { best = v; improved = true } else isDev[a] = !isDev[a]
                }
            }
            if (!improved) return Split(devRatio, isDev)
        }
        return Split(devRatio, isDev)
    }

    fun stats(items: List<EvalItem>, split: Split): List<CellStats> =
        items.indices.groupBy { items[it].quality to items[it].category }.toSortedMap(compareBy({ it.first }, { it.second }))
            .map { (key, idx) ->
                val dev = idx.filter { split.isDev[it] }
                val test = idx.filter { !split.isDev[it] }
                fun sec(l: List<Int>) = l.sumOf { items[it].gt.labelledBins } * BIN_SEC
                fun speech(l: List<Int>): Float {
                    val all = l.sumOf { items[it].gt.labelledBins }
                    return if (all == 0) Float.NaN else l.sumOf { items[it].gt.speechBins }.toFloat() / all
                }
                CellStats(key.first, key.second, dev.size, test.size, sec(dev), sec(test), speech(dev), speech(test))
            }
}
