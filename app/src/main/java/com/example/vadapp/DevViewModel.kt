package com.example.vadapp

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.provider.DocumentsContract
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.vadapp.ui.RocSeries
import com.example.vadapp.ui.modelColor
import com.example.vadapp.ui.renderRocBitmap
import com.example.vadapp.vad.ModelRegistry
import com.example.vadapp.vad.ModelSource
import com.example.vadapp.vad.NativeVad
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.SettingsStore
import com.example.vadapp.vad.VadModelType
import com.example.vadapp.vad.eval.CellStats
import com.example.vadapp.vad.eval.DatasetScanner
import com.example.vadapp.vad.eval.EvalMode
import com.example.vadapp.vad.eval.Metrics
import com.example.vadapp.vad.eval.ModelEval
import com.example.vadapp.vad.eval.ModelSummary
import com.example.vadapp.vad.eval.ParamSearch
import com.example.vadapp.vad.eval.PostConfig
import com.example.vadapp.vad.eval.SearchGrid
import com.example.vadapp.vad.eval.SearchResult
import com.example.vadapp.vad.eval.Criterion
import com.example.vadapp.vad.eval.ProbCache
import com.example.vadapp.vad.eval.ScanReport
import com.example.vadapp.vad.eval.SelectParams
import com.example.vadapp.vad.eval.Split
import com.example.vadapp.vad.eval.Splitter
import com.example.vadapp.vad.eval.XlsxReport
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

/** Per-model choices in dev mode. Settings feed configurations A2..A4. */
data class DevModelConf(
    val enabled: Boolean,
    val source: ModelSource?,
    val settings: SegmentSettings,
)

data class DevState(
    val videoTree: Uri? = null,
    val gtTree: Uri? = null,
    val scanning: Boolean = false,
    val scan: ScanReport? = null,
    val devRatio: Float = 0.7f,
    val split: Split? = null,
    val cellStats: List<CellStats> = emptyList(),
    val sources: Map<VadModelType, List<ModelSource>> = emptyMap(),
    val models: Map<VadModelType, DevModelConf> = emptyMap(),
    /** Minimum speech fraction for a 0.5 s bin to count as speech. */
    val binRatio: Float = 0.2f,
    val busy: Boolean = false,
    val progress: Float = 0f,
    val progressText: String = "",
    val evals: List<ModelEval> = emptyList(),
    /** True when split / ratio / settings changed after [evals] was computed. */
    val stale: Boolean = false,
    val params: SelectParams = SelectParams(),
    val summaries: List<ModelSummary> = emptyList(),
    /** Grid-search merge/drop/pad on dev before the sweep (config A5). */
    val searchEnabled: Boolean = false,
    val grid: SearchGrid = SearchGrid(),
    val message: String? = null,
    val error: String? = null,
)

class DevViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(DevState())
    val state: StateFlow<DevState> = _state.asStateFlow()

    private val prefs = app.getSharedPreferences("dev_mode", Application.MODE_PRIVATE)
    private val registry = ModelRegistry(app)
    private val cache = ProbCache(app.filesDir)
    private var job: Job? = null

    init {
        val user = SettingsStore(app)
        val sources = VadModelType.entries.associateWith { registry.sources(it) }
        _state.update {
            it.copy(
                videoTree = prefs.getString("videoTree", null)?.let(Uri::parse),
                gtTree = prefs.getString("gtTree", null)?.let(Uri::parse),
                sources = sources,
                models = VadModelType.entries.associateWith { m ->
                    DevModelConf(sources[m].orEmpty().isNotEmpty(), sources[m]?.firstOrNull(), user.load(m))
                },
                searchEnabled = prefs.getBoolean("searchEnabled", false),
                grid = loadGrid(),
            )
        }
    }

    // ------------------------------------------------------------------ data

    fun setVideoTree(uri: Uri) = setTree("videoTree", uri) { copy(videoTree = uri) }
    fun setGtTree(uri: Uri) = setTree("gtTree", uri) { copy(gtTree = uri) }

    private fun setTree(key: String, uri: Uri, apply: DevState.() -> DevState) {
        runCatching {
            getApplication<Application>().contentResolver
                .takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        prefs.edit().putString(key, uri.toString()).apply()
        _state.update { it.apply().copy(scan = null, split = null, cellStats = emptyList(), evals = emptyList(), summaries = emptyList()) }
    }

    fun scan() {
        val s = _state.value
        val v = s.videoTree ?: return
        val g = s.gtTree ?: return
        launchBusy("Đang quét thư mục…") {
            val report = withContext(Dispatchers.IO) {
                DatasetScanner.scan(getApplication<Application>().contentResolver, v, g)
            }
            _state.update { it.copy(scan = report, evals = emptyList(), summaries = emptyList()) }
            makeSplit()
        }
    }

    fun setDevRatio(r: Float) {
        _state.update { it.copy(devRatio = r) }
    }

    fun makeSplit() {
        val s = _state.value
        val items = s.scan?.items ?: return
        val split = Splitter.split(items, s.devRatio)
        _state.update {
            it.copy(split = split, cellStats = Splitter.stats(items, split), stale = it.evals.isNotEmpty())
        }
    }

    // ------------------------------------------------------------------ models

    fun setModelEnabled(m: VadModelType, enabled: Boolean) = updateModel(m) { copy(enabled = enabled) }
    fun setModelSource(m: VadModelType, src: ModelSource) = updateModel(m) { copy(source = src) }
    fun setModelSettings(m: VadModelType, st: SegmentSettings) = updateModel(m) { copy(settings = st) }

    private fun updateModel(m: VadModelType, f: DevModelConf.() -> DevModelConf) {
        _state.update { s ->
            val cur = s.models[m] ?: return@update s
            s.copy(models = s.models + (m to cur.f()), stale = s.evals.isNotEmpty())
        }
    }

    fun setBinRatio(r: Float) {
        _state.update { it.copy(binRatio = r, stale = it.evals.isNotEmpty()) }
    }

    fun setSearchEnabled(enabled: Boolean) {
        prefs.edit().putBoolean("searchEnabled", enabled).apply()
        _state.update { it.copy(searchEnabled = enabled, stale = it.evals.isNotEmpty()) }
    }

    fun setGrid(g: SearchGrid) {
        prefs.edit()
            .putString("grid.merge", SearchGrid.format(g.merge))
            .putString("grid.drop", SearchGrid.format(g.drop))
            .putString("grid.pre", SearchGrid.format(g.prePad))
            .putString("grid.post", SearchGrid.format(g.postPad))
            .putInt("grid.thrStep", g.thrStep)
            .apply()
        _state.update { it.copy(grid = g, stale = it.evals.isNotEmpty() && it.searchEnabled) }
    }

    private fun loadGrid(): SearchGrid {
        val d = SearchGrid()
        fun list(key: String, def: List<Int>) = prefs.getString(key, null)?.let(SearchGrid::parse)?.ifEmpty { null } ?: def
        return SearchGrid(
            merge = list("grid.merge", d.merge),
            drop = list("grid.drop", d.drop),
            prePad = list("grid.pre", d.prePad),
            postPad = list("grid.post", d.postPad),
            thrStep = prefs.getInt("grid.thrStep", d.thrStep),
        )
    }

    fun setParams(p: SelectParams) {
        _state.update { s ->
            val items = s.scan?.items.orEmpty()
            s.copy(params = p, summaries = s.evals.map { ModelSummary.compute(it, items, p) })
        }
    }

    fun clearCache() {
        cache.clear()
        _state.update { it.copy(message = "Đã xóa cache kết quả model") }
    }

    fun cancel() {
        job?.cancel()
    }

    fun dismissMessage() = _state.update { it.copy(message = null, error = null) }

    /** Runs the selected models on every file (using the cache), then sweeps thresholds. */
    fun run() {
        val s = _state.value
        val items = s.scan?.items ?: return
        val split = s.split ?: return
        val models = s.models.filter { it.value.enabled && it.value.source != null }
        if (models.isEmpty()) {
            _state.update { it.copy(error = "Chưa chọn model nào có file") }
            return
        }
        val app = getApplication<Application>()
        launchBusy("Chạy model…") {
            val total = items.size * models.size
            val done = AtomicInteger()
            val failed = ArrayList<String>()

            // 1. Model outputs (cached on disk). Sequential: the runners already use 4 threads.
            for ((m, conf) in models) {
                val src = conf.source!!
                items.forEachIndexed { i, item ->
                    currentCoroutineContextActive()
                    if (!cache.has(item.video, m, src)) {
                        progress(done.get().toFloat() / total, "${m.label}: ${i + 1}/${items.size} ${item.name}")
                        try {
                            val job = coroutineContext[Job]
                            val r = withContext(Dispatchers.Default) {
                                NativeVad.analyzeUri(app, item.video.uri, m, src) { p ->
                                    progress((done.get() + p) / total, "${m.label}: ${i + 1}/${items.size} ${item.name}")
                                    job?.isActive != false
                                }
                            }
                            cache.put(item.video, m, src, r)
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            failed += "${m.label} · ${item.name}: ${e.message}"
                        }
                    }
                    done.incrementAndGet()
                }
            }

            // 2. Optional grid search of merge/drop/pad on dev (config A5).
            val params = _state.value.params
            val devIdx = split.devIndices()
            val searches = HashMap<VadModelType, SearchResult>()
            if (s.searchEnabled && s.grid.size > 0) {
                for ((mi, entry) in models.entries.withIndex()) {
                    val (m, conf) = entry
                    val src = conf.source!!
                    val label = "Tìm tham số ${m.label} (${mi + 1}/${models.size}): ${s.grid.size} tổ hợp × " +
                        "${s.grid.thresholds.size} ngưỡng"
                    progress(0f, label)
                    val started = System.currentTimeMillis()
                    val job = coroutineContext[Job]
                    searches[m] = ParamSearch.search(
                        devFiles = devIdx.size,
                        load = { f -> cache.get(items[devIdx[f]].video, m, src)?.let { it to items[devIdx[f]].gt } },
                        segmenter = { r, st -> NativeVad.segments(m, r, st) },
                        grid = s.grid, ratio = s.binRatio,
                        criterion = params.criterion, missWeight = params.missWeight, mrMax = params.mrMax,
                        keepGoing = { job?.isActive != false },
                        onFileDone = { n ->
                            val elapsed = System.currentTimeMillis() - started
                            val eta = elapsed / n * (devIdx.size - n) / 1000
                            progress(n.toFloat() / devIdx.size, "$label · $n/${devIdx.size} file · còn ~${fmtDuration(eta)}")
                        },
                    )
                }
            }

            // 3. Threshold sweeps of A0..A5 (cheap per file, run in parallel).
            val evals = ArrayList<ModelEval>()
            val sweepTotal = items.size * models.size
            val swept = AtomicInteger()
            for ((m, conf) in models) {
                val src = conf.source!!
                val search = searches[m]
                val tuned = search?.best?.settings ?: conf.settings
                val videos = coroutineScope {
                    items.map { item ->
                        async(Dispatchers.Default) {
                            val r = cache.get(item.video, m, src) ?: return@async null
                            val job = coroutineContext[Job]
                            Metrics.evaluateVideo(m, r, item.gt, conf.settings, tuned, s.binRatio) { job?.isActive != false }
                                .also {
                                    val n = swept.incrementAndGet()
                                    progress(n.toFloat() / sweepTotal, "Quét ngưỡng ${m.label}: $n/$sweepTotal")
                                }
                        }
                    }.awaitAll()
                }
                evals += ModelEval(m, src, conf.settings, tuned, search, s.binRatio, videos, devIdx, split.testIndices())
            }
            // After a search, report the tuned configuration by default.
            if (searches.isNotEmpty()) _state.update { it.copy(params = it.params.copy(config = PostConfig.A5)) }
            val summaries = evals.map { ModelSummary.compute(it, items, _state.value.params) }
            _state.update {
                it.copy(
                    evals = evals, summaries = summaries, stale = false,
                    error = if (failed.isEmpty()) null else "Lỗi ${failed.size} file:\n" + failed.take(10).joinToString("\n"),
                )
            }
        }
    }

    /** Search results that were optimised for another criterion / weight than the current ones. */
    fun searchOutdated(st: DevState = _state.value): Boolean = st.evals.any { e ->
        val r = e.search ?: return@any false
        r.criterion != st.params.criterion || r.missWeight != st.params.missWeight ||
            (r.criterion == Criterion.MR_MAX && r.mrMax != st.params.mrMax)
    }

    private fun fmtDuration(sec: Long) = if (sec >= 3600) "%dh%02d".format(sec / 3600, sec / 60 % 60)
        else if (sec >= 60) "%d phút".format((sec + 30) / 60) else "$sec s"

    private suspend fun currentCoroutineContextActive() {
        if (!kotlinx.coroutines.currentCoroutineContext().isActive) throw CancellationException()
    }

    private fun progress(p: Float, text: String) {
        _state.update { it.copy(progress = p.coerceIn(0f, 1f), progressText = text) }
    }

    private fun launchBusy(text: String, block: suspend kotlinx.coroutines.CoroutineScope.() -> Unit) {
        job?.cancel()
        _state.update { it.copy(busy = true, progress = 0f, progressText = text, error = null) }
        job = viewModelScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                _state.update { it.copy(message = "Đã hủy") }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message ?: e.toString()) }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    // ------------------------------------------------------------------ export

    fun rocSeries(set: Int, mode: EvalMode): List<RocSeries> {
        val s = _state.value
        val c = s.params.config.ordinal
        return s.summaries.map { ms ->
            val e = ms.eval
            val sweep = if (set == 0) e.devSweep else e.testSweep
            val auc = ms.rows[c].let { if (set == 0) it.aucDev else it.aucTest }[mode.ordinal]
            val marker = if (ms.usedK >= 0) Metrics.conf(sweep, c, ms.usedK, mode.ordinal).let { x ->
                if (x.far.isNaN() || x.recall.isNaN()) null else x.far to x.recall
            } else null
            RocSeries(
                "${e.model.label} (${e.source.format.label}) AUC ${"%.3f".format(Locale.US, auc)}",
                modelColor(e.model), Metrics.roc(sweep, c, mode.ordinal), marker,
            )
        }
    }

    /** Writes the workbook and the ROC images into a user-picked folder. */
    fun export(tree: Uri) {
        val s = _state.value
        val items = s.scan?.items ?: return
        val split = s.split ?: return
        if (s.summaries.isEmpty()) return
        val app = getApplication<Application>()
        launchBusy("Đang xuất báo cáo…") {
            withContext(Dispatchers.IO) {
                val resolver = app.contentResolver
                runCatching { resolver.takePersistableUriPermission(tree, Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION) }
                val parent = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
                val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date())
                val date = SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.US).format(Date())
                fun create(mime: String, name: String) =
                    DocumentsContract.createDocument(resolver, parent, mime, name) ?: error("Không tạo được $name")

                val xlsx = create("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vad_dev_report_$stamp.xlsx")
                resolver.openOutputStream(xlsx)!!.use { out ->
                    XlsxReport(
                        items, split, s.summaries, s.params, s.binRatio,
                        s.videoTree?.let(DatasetScanner::treeName).orEmpty(),
                        s.gtTree?.let(DatasetScanner::treeName).orEmpty(), date,
                    ).write(out)
                }
                for (set in 0..1) for (mode in EvalMode.entries) {
                    val setName = if (set == 0) "dev" else "test"
                    val bmp = renderRocBitmap("ROC $setName · ${mode.label} · ${s.params.config.code}", rocSeries(set, mode))
                    val png = create("image/png", "roc_${stamp}_${setName}_${mode.short}.png")
                    resolver.openOutputStream(png)!!.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    bmp.recycle()
                }
            }
            _state.update { it.copy(message = "Đã xuất vad_dev_report_*.xlsx và 6 ảnh ROC") }
        }
    }
}
