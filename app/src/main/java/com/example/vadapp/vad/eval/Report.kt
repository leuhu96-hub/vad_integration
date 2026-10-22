package com.example.vadapp.vad.eval

import com.example.vadapp.vad.ModelSource
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.VadModelType

/** Output of the expensive part (model run + threshold sweep) for one model. */
class ModelEval(
    val model: VadModelType,
    val source: ModelSource,
    /** Segmenter settings A2..A4 were built from. */
    val settings: SegmentSettings,
    /** merge/drop/pad of A5: the grid-search optimum, or [settings] when no search ran. */
    val tuned: SegmentSettings,
    val search: SearchResult?,
    val binRatio: Float,
    /** Aligned with the dataset items; null when the file could not be analysed. */
    val videos: List<VideoEval?>,
    val devIdx: List<Int>,
    val testIdx: List<Int>,
) {
    val devSweep: IntArray = Metrics.sum(devIdx.mapNotNull { videos[it] })
    val testSweep: IntArray = Metrics.sum(testIdx.mapNotNull { videos[it] })
    val exactAucDev = DoubleArray(EvalMode.entries.size) { Metrics.exactAuc(devIdx.mapNotNull { i -> videos[i] }, it) }
    val exactAucTest = DoubleArray(EvalMode.entries.size) { Metrics.exactAuc(testIdx.mapNotNull { i -> videos[i] }, it) }
    val failed: Int get() = videos.count { it == null }

    /** Description of [cfg] with the values actually used (A5 shows the tuned parameters). */
    fun describe(cfg: PostConfig): String = when (cfg) {
        PostConfig.A5 -> if (search?.best == null) "A5 = A4 (chưa tìm tham số)" else
            "Tối ưu trên dev: merge ${tuned.minSilenceMs}, drop ${tuned.minSpeechMs}, pad ${tuned.prePadMs}/${tuned.postPadMs} ms"
        else -> cfg.description
    }
}

/** Threshold selection / reporting parameters: cheap to change, no re-sweep needed. */
data class SelectParams(
    val missWeight: Double = 0.75,
    val config: PostConfig = PostConfig.A4,
    val criterion: Criterion = Criterion.DCF_MIN,
    val fpr1: Double = 0.01,
    val fpr2: Double = 0.05,
    val mrMax: Double = 0.05,
)

/** Table A row: threshold choice on dev (strict) for one configuration. */
class ConfigRow(
    val config: PostConfig,
    val thrDcf: Int, val dcfMin: Double,
    val thrF1: Int, val f1Max: Double,
    val thrYouden: Int, val youdenMax: Double,
    val thrEer: Int, val thrMr: Int,
    val aucDev: DoubleArray, val aucTest: DoubleArray,
    /** Only for A0 (threshold-free), NaN otherwise. */
    val exactAucDev: DoubleArray, val exactAucTest: DoubleArray,
    val tprFpr1: Double, val tprFpr2: Double,
    val used: Int,
)

class GroupRow(val kind: String, val name: String, val files: Int, val conf: Conf)

class ModelSummary(
    val eval: ModelEval,
    val rows: List<ConfigRow>,
    val config: PostConfig,
    /** Threshold index used for the test tables (-1 = none). */
    val usedK: Int,
    /** Test confusion per mode at (config, usedK). */
    val test: List<Conf>,
    /** Per mode: CI95 for F1, DCF, MR, FAR (null if not computable). */
    val ci: List<Map<String, Pair<Double, Double>?>>,
    val groups: List<GroupRow>,
) {
    val model get() = eval.model
    val threshold: Float? get() = if (usedK >= 0) thresholdAt(usedK) else null
    /** Same threshold as a double without float rounding noise (for the workbook). */
    val thresholdExact: Double? get() = if (usedK >= 0) usedK / 100.0 else null

    /** Settings that reproduce [config] in the user's pipeline (A0 has no segmenter equivalent). */
    fun applicableSettings(): SegmentSettings? {
        val t = threshold ?: return null
        if (config == PostConfig.A0) return null
        return config.settings(eval.settings, eval.tuned, t)
    }

    companion object {
        val CI_METRICS = listOf("F1", "DCF", "MR", "FAR")

        fun compute(e: ModelEval, items: List<EvalItem>, p: SelectParams): ModelSummary {
            val strict = EvalMode.STRICT.ordinal
            val rows = PostConfig.entries.map { cfg ->
                val c = cfg.ordinal
                fun sel(cr: Criterion) = Metrics.select(e.devSweep, c, strict, cr, p.missWeight, p.mrMax)
                val kDcf = sel(Criterion.DCF_MIN)
                val kF1 = sel(Criterion.F1_MAX)
                val kJ = sel(Criterion.YOUDEN)
                fun at(k: Int) = if (k < 0) null else Metrics.conf(e.devSweep, c, k, strict)
                ConfigRow(
                    config = cfg,
                    thrDcf = kDcf, dcfMin = at(kDcf)?.dcf(p.missWeight) ?: Double.NaN,
                    thrF1 = kF1, f1Max = at(kF1)?.f1 ?: Double.NaN,
                    thrYouden = kJ, youdenMax = at(kJ)?.let { it.recall - it.far } ?: Double.NaN,
                    thrEer = sel(Criterion.EER), thrMr = sel(Criterion.MR_MAX),
                    aucDev = DoubleArray(3) { Metrics.auc(Metrics.roc(e.devSweep, c, it)) },
                    aucTest = DoubleArray(3) { Metrics.auc(Metrics.roc(e.testSweep, c, it)) },
                    exactAucDev = if (cfg == PostConfig.A0) e.exactAucDev else DoubleArray(3) { Double.NaN },
                    exactAucTest = if (cfg == PostConfig.A0) e.exactAucTest else DoubleArray(3) { Double.NaN },
                    tprFpr1 = Metrics.tprAtFar(e.devSweep, c, strict, p.fpr1),
                    tprFpr2 = Metrics.tprAtFar(e.devSweep, c, strict, p.fpr2),
                    used = sel(p.criterion),
                )
            }
            val c = p.config.ordinal
            val k = rows[c].used
            val test = EvalMode.entries.map { m ->
                if (k < 0) Conf.ZERO else Metrics.conf(e.testSweep, c, k, m.ordinal)
            }
            val ci = EvalMode.entries.map { m ->
                if (k < 0) emptyMap() else {
                    val per = e.testIdx.mapNotNull { e.videos[it]?.conf(c, k, m.ordinal) }
                    mapOf(
                        "F1" to Bootstrap.ci(per, { it.f1 }),
                        "DCF" to Bootstrap.ci(per, { it.dcf(p.missWeight) }),
                        "MR" to Bootstrap.ci(per, { it.mr }),
                        "FAR" to Bootstrap.ci(per, { it.far }),
                    )
                }
            }
            val groups = if (k < 0) emptyList() else buildList {
                fun group(kind: String, key: (EvalItem) -> String) {
                    e.testIdx.groupBy { key(items[it]) }.toSortedMap().forEach { (name, idx) ->
                        val conf = idx.mapNotNull { e.videos[it]?.conf(c, k, strict) }.fold(Conf.ZERO, Conf::plus)
                        add(GroupRow(kind, name, idx.size, conf))
                    }
                }
                group("Quality") { it.quality }
                group("Category") { it.category }
            }
            return ModelSummary(e, rows, p.config, k, test, ci, groups)
        }
    }
}
