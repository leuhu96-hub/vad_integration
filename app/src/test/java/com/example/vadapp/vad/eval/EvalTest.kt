package com.example.vadapp.vad.eval

import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.ModelFormat
import com.example.vadapp.vad.ModelSource
import com.example.vadapp.vad.Segment
import com.example.vadapp.vad.SegmentSettings
import kotlinx.coroutines.runBlocking
import com.example.vadapp.vad.VadModelType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Random
import kotlin.math.abs

class EvalTest {

    private fun result(probs: FloatArray, hop: Float, duration: Float) =
        AnalysisResult(probs, hop, 0f, duration, 16000, 1, "", "ONNX", 0, FloatArray(0), 1f)

    /** Frames with p >= t become segments; A4 adds 100 ms of padding on both sides. */
    private fun fakeSegmenter(r: AnalysisResult): (PostConfig, Float) -> List<Segment> = { cfg, t ->
        val out = ArrayList<Segment>()
        var start = -1
        for (i in 0..r.probs.size) {
            val on = i < r.probs.size && r.probs[i] >= t
            if (on && start < 0) start = i
            if (!on && start >= 0) {
                val pad = if (cfg == PostConfig.A4) 0.1f else 0f
                out += Segment(maxOf(0f, start * r.hopSec - pad), minOf(r.durationSec, i * r.hopSec + pad))
                start = -1
            }
        }
        out
    }

    @Test
    fun parsesGroundTruth() {
        val gt = GroundTruth.parse("0 0.5 speech\n0.5 1 no-speech\n1\t1.5\tSpeech\n\n2.0 2.5 no-speech\n")
        assertEquals(5, gt.bins)
        assertEquals(listOf<Byte>(1, 0, 1, -1, 0), gt.labels.toList())
        assertEquals(2, gt.speechBins)
        assertEquals(4, gt.labelledBins)
    }

    @Test
    fun countsBinsWithRatioAndModes() {
        // 4 bins: speech, speech, non, non. 10 ms frames.
        val gt = GroundTruth(byteArrayOf(1, 1, 0, 0))
        val probs = FloatArray(200)
        for (i in 0 until 10) probs[i] = 0.9f          // bin 0: 0.1 s of speech = 20 %
        for (i in 50 until 100) probs[i] = 0.9f        // bin 1: fully speech
        for (i in 100 until 105) probs[i] = 0.9f       // bin 2: 0.05 s = 10 %
        val r = result(probs, 0.01f, 2.0f)
        val v = Metrics.evaluate(r, gt, 0.2f, segmenter = fakeSegmenter(r))
        // Boundary bins are 1 and 2 (the speech -> non-speech change).
        assertEquals(listOf(false, true, true, false), v.boundary.toList())
        val k = 50 // t = 0.5
        for (c in listOf(0, 1)) {
            val strict = v.conf(c, k, 0)
            assertEquals("config $c", Conf(2, 0, 0, 2), strict)
            assertEquals(Conf(1, 0, 0, 1), v.conf(c, k, 1))
            assertEquals(Conf(1, 0, 0, 1), v.conf(c, k, 2))
        }
        // A4 pads bin 2 to 0.15 s (30 %) -> false alarm there.
        assertEquals(Conf(2, 0, 1, 1), v.conf(4, k, 0))
        // t = 0: every bin speech.
        assertEquals(Conf(2, 0, 2, 0), v.conf(0, 0, 0))
    }

    @Test
    fun exactAucMatchesPairCount() {
        val rnd = Random(1)
        val labels = ByteArray(300) { if (rnd.nextBoolean()) 1 else 0 }
        val scores = FloatArray(300) { (labels[it] * 0.3f + rnd.nextInt(10) / 10f) }
        val v = VideoEval(IntArray(SWEEP_SIZE), scores, labels, BooleanArray(300))
        var wins = 0.0
        var pairs = 0
        for (i in labels.indices) for (j in labels.indices) {
            if (labels[i].toInt() == 1 && labels[j].toInt() == 0) {
                pairs++
                wins += if (scores[i] > scores[j]) 1.0 else if (scores[i] == scores[j]) 0.5 else 0.0
            }
        }
        assertEquals(wins / pairs, Metrics.exactAuc(listOf(v), 0), 1e-9)
    }

    @Test
    fun splitIsStratifiedAndBalanced() {
        val items = dataset(Random(7), 90)
        val split = Splitter.split(items, 0.7f)
        val stats = Splitter.stats(items, split)
        for (c in stats) {
            if (c.devCount + c.testCount >= 2) assertTrue("${c.quality}×${c.category}", c.devCount > 0 && c.testCount > 0)
        }
        val devSec = stats.sumOf { it.devSec.toDouble() }
        val allSec = stats.sumOf { (it.devSec + it.testSec).toDouble() }
        assertEquals(0.7, devSec / allSec, 0.03)
    }

    /**
     * Synthetic end-to-end run: writes the workbook plus the values the app shows, for
     * tools/check_xlsx.py to compare against the workbook's own formulas.
     */
    @Test
    fun writesWorkbook() {
        val rnd = Random(3)
        val items = dataset(rnd, 30)
        val split = Splitter.split(items, 0.7f)
        val models = listOf(VadModelType.FIRERED, VadModelType.SILERO)
        val dev = split.devIndices()
        val evals = models.mapIndexed { mi, m ->
            val results = items.map { item ->
                val n = item.gt.bins
                val probs = FloatArray(n * 50) { j ->
                    val l = item.gt.labels[j / 50]
                    val base = if (l.toInt() == 1) 0.55f + 0.1f * mi else 0.25f
                    (base + (rnd.nextGaussian() * 0.2).toFloat()).coerceIn(0f, 1f)
                }
                result(probs, 0.01f, n * BIN_SEC)
            }
            // The first model gets a (small) parameter search, so A5 and its sheet are exercised.
            val search = if (mi == 0) runBlocking {
                ParamSearch.search(
                    dev.size, { results[dev[it]] to items[dev[it]].gt }, ::settingsSegmenter,
                    SearchGrid(listOf(0, 200), listOf(0, 100), listOf(0, 50), listOf(0, 100), thrStep = 5),
                    0.2f, Criterion.DCF_MIN, 0.75, 0.05,
                )
            } else null
            val tuned = search?.best?.settings ?: m.defaults
            val videos = items.indices.map { i ->
                Metrics.evaluate(results[i], items[i].gt, 0.2f) { cfg, t ->
                    settingsSegmenter(results[i], cfg.settings(m.defaults, tuned, t))
                }
            }
            ModelEval(
                m, ModelSource("asset:models/x.onnx", ModelFormat.ONNX, "${m.assetBase}.onnx", false),
                m.defaults, tuned, search, 0.2f, videos, dev, split.testIndices(),
            )
        }
        val params = SelectParams(config = PostConfig.A5)
        val summaries = evals.map { ModelSummary.compute(it, items, params) }
        val dir = File("build/eval-test").apply { mkdirs() }
        File(dir, "report.xlsx").outputStream().use {
            XlsxReport(items, split, summaries, params, 0.2f, "phase1", "gt", "test").write(it)
        }
        fun num(d: Double) = if (d.isNaN()) "null" else d.toString()
        val expected = summaries.joinToString(",", "[", "]") { ms ->
            val rows = ms.rows.joinToString(",", "[", "]") { r ->
                "{\"thrDcf\":${r.thrDcf},\"dcfMin\":${num(r.dcfMin)},\"thrF1\":${r.thrF1},\"f1Max\":${num(r.f1Max)}," +
                    "\"thrYouden\":${r.thrYouden},\"thrEer\":${r.thrEer},\"thrMr\":${r.thrMr}," +
                    "\"tpr1\":${num(r.tprFpr1)},\"tpr2\":${num(r.tprFpr2)},\"used\":${r.used}}"
            }
            val test = ms.test.joinToString(",", "[", "]") { c -> "[${c.tp},${c.fn},${c.fp},${c.tn}]" }
            "{\"model\":\"${ms.model.label}\",\"usedK\":${ms.usedK},\"rows\":$rows,\"test\":$test}"
        }
        File(dir, "expected.json").writeText(expected)
        assertTrue(summaries.all { it.usedK >= 0 })
        // Sanity: AUC of separable synthetic data is well above chance.
        assertTrue(summaries.all { it.rows[0].aucDev[0] > 0.7 && abs(it.rows[0].aucDev[0] - it.rows[0].exactAucDev[0]) < 0.05 })
    }

    /** Threshold, then pad, merge short gaps and drop short segments, like a real segmenter. */
    private fun settingsSegmenter(r: AnalysisResult, st: SegmentSettings): List<Segment> {
        val raw = fakeSegmenter(r)(PostConfig.A1, st.threshold)
        val padded = raw.map { Segment(maxOf(0f, it.startSec - st.prePadMs / 1000f), minOf(r.durationSec, it.endSec + st.postPadMs / 1000f)) }
        val merged = ArrayList<Segment>()
        for (sg in padded) {
            val last = merged.lastOrNull()
            if (last != null && (sg.startSec - last.endSec) * 1000 < st.minSilenceMs) {
                merged[merged.size - 1] = Segment(last.startSec, maxOf(last.endSec, sg.endSec))
            } else merged += sg
        }
        return merged.filter { it.durationSec * 1000 >= st.minSpeechMs }
    }

    @Test
    fun searchMatchesBruteForce() = runBlocking {
        val rnd = Random(11)
        val items = dataset(rnd, 8)
        // Noisy probabilities with dropouts inside speech, so merge and padding matter.
        val results = items.map { item ->
            val n = item.gt.bins
            val probs = FloatArray(n * 50) { j ->
                val speech = item.gt.labels[j / 50].toInt() == 1
                val p = if (speech) (if (rnd.nextInt(5) == 0) 0.1f else 0.8f) else (if (rnd.nextInt(20) == 0) 0.7f else 0.2f)
                (p + (rnd.nextGaussian() * 0.1).toFloat()).coerceIn(0f, 1f)
            }
            result(probs, 0.01f, n * BIN_SEC)
        }
        val grid = SearchGrid(listOf(0, 300), listOf(0, 200), listOf(0, 100), listOf(0, 150), thrStep = 10)
        val w = 0.75
        val res = ParamSearch.search(
            devFiles = items.size, load = { results[it] to items[it].gt },
            segmenter = ::settingsSegmenter, grid = grid, ratio = 0.2f,
            criterion = Criterion.DCF_MIN, missWeight = w, mrMax = 0.05,
        )
        assertEquals(16, res.entries.size)
        val thr = grid.thresholds
        val a5 = PostConfig.A5.ordinal
        // Brute force: full evaluation with each combination as the tuned settings (A5).
        res.entries.forEachIndexed { c, e ->
            val combo = grid.combos()[c]
            val sweep = Metrics.sum(items.indices.map { i ->
                Metrics.evaluate(results[i], items[i].gt, 0.2f) { cfg, t ->
                    settingsSegmenter(results[i], cfg.settings(combo, combo, t))
                }
            })
            val (ti, cost) = Metrics.selectBy(thr.size, { Metrics.conf(sweep, a5, thr[it], 0) }, Criterion.DCF_MIN, w, 0.05)
            assertEquals("combo $c threshold", thr[ti], e.bestK)
            assertEquals("combo $c cost", cost, e.cost, 1e-12)
            assertEquals("combo $c conf", Metrics.conf(sweep, a5, thr[ti], 1), e.conf[1])
        }
        val best = res.best!!
        assertEquals(res.entries.filter { !it.cost.isNaN() }.minOf { it.cost }, best.cost, 0.0)
        // With the tuned settings, A5 on the full 0.01 grid is at least as good as the coarse search.
        val tuned = best.settings
        val sweep = Metrics.sum(items.indices.map { i ->
            Metrics.evaluate(results[i], items[i].gt, 0.2f) { cfg, t -> settingsSegmenter(results[i], cfg.settings(tuned, tuned, t)) }
        })
        val full = Metrics.selectBy(NUM_THR, { Metrics.conf(sweep, a5, it, 0) }, Criterion.DCF_MIN, w, 0.05).second
        assertTrue(full <= best.cost + 1e-12)
        // Marginals: the best value over a parameter's values is the global best.
        for (p in 0..3) assertEquals(res.value(best), res.marginal(p).minOf { it.second }, 1e-12)
        // FireRedVAD smoothing axis: outermost; parameters the grid does not search come from base.
        val base = VadModelType.FIRERED.defaults.copy(maxSpeechSec = 7f)
        val smoothCombos = grid.combos(base, withSmooth = true)
        assertEquals(grid.size(withSmooth = true), smoothCombos.size)
        assertEquals(grid.smooth, smoothCombos.map { it.smoothWindowMs }.distinct())
        assertTrue(smoothCombos.all { it.maxSpeechSec == 7f })
        assertTrue(grid.combos(base).all { it.smoothWindowMs == base.smoothWindowMs })
    }

    /** Test-only mode: every file is test, thresholds fixed per model; writes a workbook for check_xlsx.py. */
    @Test
    fun testOnlyUsesFixedThreshold() {
        val rnd = Random(5)
        val items = dataset(rnd, 20)
        val split = Split(0f, BooleanArray(items.size))
        val models = listOf(VadModelType.FIRERED, VadModelType.SILERO)
        val fixed = listOf(37, 62)
        val summaries = models.mapIndexed { mi, m ->
            val videos = items.map { item ->
                val n = item.gt.bins
                val probs = FloatArray(n * 50) { j ->
                    val base = if (item.gt.labels[j / 50].toInt() == 1) 0.6f else 0.3f
                    (base + (rnd.nextGaussian() * 0.2).toFloat()).coerceIn(0f, 1f)
                }
                val r = result(probs, 0.01f, n * BIN_SEC)
                Metrics.evaluate(r, item.gt, 0.2f) { cfg, t -> settingsSegmenter(r, cfg.settings(m.defaults, m.defaults, t)) }
            }
            val e = ModelEval(
                m, ModelSource("asset:models/x.onnx", ModelFormat.ONNX, "${m.assetBase}.onnx", false),
                m.defaults, m.defaults, null, 0.2f, videos, split.devIndices(), split.testIndices(),
            )
            val ms = ModelSummary.compute(e, items, SelectParams(), fixed[mi])
            // The test tables use the fixed threshold for config A4 over all files.
            assertEquals(fixed[mi], ms.usedK)
            assertEquals(items.size, e.testIdx.size)
            val all = videos.map { it.conf(PostConfig.A4.ordinal, fixed[mi], 0) }.fold(Conf.ZERO, Conf::plus)
            assertEquals(all, ms.test[0])
            assertTrue(ms.rows[PostConfig.A4.ordinal].aucTest[0] > 0.5)
            ms
        }
        val dir = File("build/eval-test/test-only").apply { mkdirs() }
        File(dir, "report.xlsx").outputStream().use {
            XlsxReport(items, split, summaries, SelectParams(), 0.2f, "phase1", "gt", "test", testOnly = true).write(it)
        }
        val expected = summaries.joinToString(",", "[", "]") { ms ->
            val test = ms.test.joinToString(",", "[", "]") { c -> "[${c.tp},${c.fn},${c.fp},${c.tn}]" }
            "{\"model\":\"${ms.model.label}\",\"testOnly\":true,\"nConfigs\":${ms.rows.size},\"usedK\":${ms.usedK},\"test\":$test}"
        }
        File(dir, "expected.json").writeText(expected)
    }

    private fun dataset(rnd: Random, n: Int): List<EvalItem> = List(n) { i ->
        val bins = 40 + rnd.nextInt(200)
        val labels = ByteArray(bins)
        var cur: Byte = 0
        for (b in 0 until bins) {
            if (rnd.nextInt(6) == 0) cur = (1 - cur).toByte()
            labels[b] = cur
        }
        val q = listOf("A", "B", "C")[i % 3]
        val c = listOf("M", "N", "Q")[(i / 3) % 3]
        val name = "v%03d.mp4".format(i)
        EvalItem(
            TreeFile("content://v/$i", name, "$q/sub/$name", q, 1, 0),
            TreeFile("content://g/$i", name.replace(".mp4", ".txt"), "$c/${name.replace(".mp4", ".txt")}", c, 1, 0),
            GroundTruth(labels),
        )
    }
}
