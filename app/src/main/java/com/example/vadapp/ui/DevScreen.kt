package com.example.vadapp.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.vadapp.DevState
import com.example.vadapp.DevViewModel
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.VadModelType
import com.example.vadapp.vad.eval.BIN_SEC
import com.example.vadapp.vad.eval.Criterion
import com.example.vadapp.vad.eval.DatasetScanner
import com.example.vadapp.vad.eval.EvalMode
import com.example.vadapp.vad.eval.ModelSummary
import com.example.vadapp.vad.eval.PostConfig
import com.example.vadapp.vad.eval.Metrics
import com.example.vadapp.vad.eval.NUM_THR
import com.example.vadapp.vad.eval.SearchGrid
import com.example.vadapp.vad.eval.SearchResult
import java.util.Locale
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DevScreen(
    vm: DevViewModel,
    onApply: (VadModelType, SegmentSettings) -> Unit,
    onBack: () -> Unit,
) {
    val st by vm.state.collectAsStateWithLifecycle()
    BackHandler(onBack = onBack)

    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { it?.let(vm::setVideoTree) }
    val gtPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { it?.let(vm::setGtTree) }
    val exportPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { it?.let(vm::export) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Chế độ Dev · chọn tham số") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Quay lại") }
                },
                actions = {
                    IconButton(onClick = vm::clearCache, enabled = !st.busy) {
                        Icon(Icons.Filled.DeleteSweep, contentDescription = "Xóa cache kết quả model")
                    }
                },
            )
        },
        bottomBar = {
            if (st.busy) {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                    Text(st.progressText, style = MaterialTheme.typography.labelSmall, maxLines = 1)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        LinearProgressIndicator(progress = { st.progress }, modifier = Modifier.weight(1f))
                        Text(" ${(st.progress * 100).roundToInt()}%", style = MaterialTheme.typography.labelMedium)
                        TextButton(onClick = vm::cancel) { Text("Hủy") }
                    }
                }
            }
        },
    ) { padding ->
        LazyColumn(
            Modifier.padding(padding).fillMaxSize(),
            contentPadding = PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(key = "data") {
                Section("1. Dữ liệu") {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { videoPicker.launch(st.videoTree) }, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Filled.FolderOpen, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text(st.videoTree?.let(DatasetScanner::treeName) ?: "Thư mục video", maxLines = 1)
                        }
                        OutlinedButton(onClick = { gtPicker.launch(st.gtTree) }, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Filled.FolderOpen, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text(st.gtTree?.let(DatasetScanner::treeName) ?: "Thư mục GT", maxLines = 1)
                        }
                    }
                    Hint("Video: <gốc>/<quality>/…/tên.mp4 · GT: <gốc>/<category>/…/tên.txt (dòng \"0 0.5 speech\"). Ghép theo tên file.")
                    Button(onClick = vm::scan, enabled = st.videoTree != null && st.gtTree != null && !st.busy) {
                        Text("Quét & ghép")
                    }
                    st.scan?.let { r ->
                        Text(
                            "${r.videoCount} video · ${r.gtCount} file GT · ghép được ${r.items.size}",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Warn("Video không có GT", r.videosWithoutGt)
                        Warn("GT không có video", r.gtWithoutVideo)
                        Warn("Trùng tên (bỏ qua)", r.duplicates)
                        Warn("GT rỗng/không đọc được", r.emptyGt)
                    }
                }
            }

            if (st.scan != null) item(key = "split") { SplitSection(st, vm) }
            if (st.split != null) item(key = "models") { ModelsSection(st, vm) }

            if (st.summaries.isNotEmpty()) {
                item(key = "params") { ParamsSection(st, vm) }
                item(key = "roc") { RocSection(vm, st) }
                item(key = "compare") { CompareSection(st) }
                st.summaries.filter { it.eval.search != null }.forEach { ms ->
                    item(key = "search-${ms.model.name}") { SearchSection(ms, st, vm) }
                }
                st.summaries.forEach { ms ->
                    item(key = "sum-${ms.model.name}") { SummarySection(ms, st, onApply) }
                }
                item(key = "export") {
                    Section("Xuất báo cáo") {
                        Hint("Tạo vad_dev_report_<ngày>.xlsx (bảng tổng hợp có công thức, sheet Quét ngưỡng, Split, Theo nhóm) và " +
                            "${if (st.testOnly) 3 else 6} ảnh ROC và 1 ảnh gộp tất cả (roc_*_all.png) trong thư mục bạn chọn.")
                        Button(onClick = { exportPicker.launch(st.videoTree) }, enabled = !st.busy) { Text("Chọn thư mục & xuất") }
                    }
                }
            }
        }
    }

    val msg = st.error ?: st.message
    if (msg != null) {
        AlertDialog(
            onDismissRequest = vm::dismissMessage,
            confirmButton = { TextButton(onClick = vm::dismissMessage) { Text("OK") } },
            title = { Text(if (st.error != null) "Lỗi" else "Thông báo") },
            text = { Text(msg) },
        )
    }
}

@Composable
private fun Section(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
            content()
        }
    }
}

@Composable
private fun Hint(text: String) {
    Text(text, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun Warn(title: String, list: List<String>) {
    if (list.isEmpty()) return
    var open by rememberSaveable(title) { mutableStateOf(false) }
    TextButton(onClick = { open = !open }, contentPadding = PaddingValues(0.dp)) {
        Text("⚠ $title: ${list.size} ${if (open) "▲" else "▼"}", color = MaterialTheme.colorScheme.error)
    }
    if (open) {
        Text(
            list.take(30).joinToString("\n") + if (list.size > 30) "\n… +${list.size - 30}" else "",
            style = MaterialTheme.typography.labelSmall,
            fontFamily = FontFamily.Monospace,
        )
    }
}

@Composable
private fun SliderRow(label: String, value: String, v: Float, range: ClosedFloatingPointRange<Float>, steps: Int = 0, onChange: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.width(118.dp))
        Slider(value = v, onValueChange = onChange, valueRange = range, steps = steps, modifier = Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace,
            textAlign = TextAlign.End, modifier = Modifier.width(64.dp))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun <T> ChipRow(label: String, options: List<T>, selected: T, text: (T) -> String, onSelect: (T) -> Unit) {
    Column {
        Text(label, style = MaterialTheme.typography.labelMedium)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            options.forEach { o ->
                FilterChip(selected = o == selected, onClick = { onSelect(o) }, label = { Text(text(o)) })
            }
        }
    }
}

@Composable
private fun SplitSection(st: DevState, vm: DevViewModel) = Section("2. Chế độ & chia tập") {
    ChipRow("Chế độ", listOf(false, true), st.testOnly, { if (it) "Chỉ test (cấu hình user)" else "Chọn tham số (dev → test)" }) {
        if (!st.busy) vm.setTestOnly(it)
    }
    if (st.testOnly) {
        Hint("Không chia dev: cả ${st.scan?.items?.size ?: 0} file dùng để đánh giá cấu hình hiện tại của user " +
            "(model, merge, drop, pad) với ngưỡng bạn chọn ở mục 3.")
    } else {
        SliderRow("Tỷ lệ dev", "${(st.devRatio * 100).roundToInt()}%", st.devRatio, 0.5f..0.9f, steps = 7) {
            vm.setDevRatio((it * 20).roundToInt() / 20f)
        }
        Hint("Phân tầng theo quality × category, cân theo thời lượng rồi hoán đổi file để tỷ lệ speech hai tập gần nhau.")
        FilledTonalButton(onClick = vm::makeSplit, enabled = !st.busy) { Text("Chia lại") }
    }
    if (st.cellStats.isNotEmpty()) {
        val rows = st.cellStats.map { c ->
            listOf(
                "${c.quality.ifEmpty { "(gốc)" }} × ${c.category.ifEmpty { "(gốc)" }}",
                "${c.devCount}", "${c.testCount}",
                fmtMin(c.devSec), fmtMin(c.testSec),
                fmtPct(c.devSpeech.toDouble()), fmtPct(c.testSpeech.toDouble()),
            )
        }
        val devN = st.cellStats.sumOf { it.devCount }
        val testN = st.cellStats.sumOf { it.testCount }
        val devS = st.cellStats.sumOf { it.devSec.toDouble() }
        val testS = st.cellStats.sumOf { it.testSec.toDouble() }
        val items = st.scan?.items.orEmpty()
        val split = st.split
        fun speech(dev: Boolean): Double {
            if (split == null) return Double.NaN
            val idx = items.indices.filter { split.isDev[it] == dev }
            val all = idx.sumOf { items[it].gt.labelledBins }
            return if (all == 0) Double.NaN else idx.sumOf { items[it].gt.speechBins }.toDouble() / all
        }
        val total = listOf(
            "Tổng (dev ${fmtPct(devS / (devS + testS))} thời lượng)", "$devN", "$testN",
            fmtMin(devS.toFloat()), fmtMin(testS.toFloat()), fmtPct(speech(true)), fmtPct(speech(false)),
        )
        DataTable(
            header = listOf("Quality × Category", "Dev", "Test", "Dev (phút)", "Test (phút)", "Speech dev", "Speech test"),
            rows = rows + listOf(total),
            firstColWidth = 170.dp,
            highlightRow = rows.size,
        )
    }
}

private fun fmtMin(sec: Float) = String.format(Locale.US, "%.1f", sec / 60)

@Composable
private fun ModelsSection(st: DevState, vm: DevViewModel) = Section("3. Model & hậu xử lý") {
    VadModelType.entries.forEach { m ->
        val conf = st.models[m] ?: return@forEach
        val sources = st.sources[m].orEmpty()
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = conf.enabled, onCheckedChange = { vm.setModelEnabled(m, it) }, enabled = sources.isNotEmpty())
            Text(m.label, style = MaterialTheme.typography.titleSmall)
            if (sources.isEmpty()) Text("  (không có file model)", color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.labelSmall)
        }
        if (conf.enabled && sources.isNotEmpty()) {
            Column(Modifier.padding(start = 12.dp)) {
                if (sources.size > 1) {
                    ChipRow("File model", sources, conf.source, { if (it?.custom == true) it.fileName else it?.label.orEmpty() }) {
                        if (it != null) vm.setModelSource(m, it)
                    }
                }
                val s = conf.settings
                if (st.testOnly) {
                    SliderRow("Ngưỡng", String.format(Locale.US, "%.2f", s.threshold), s.threshold, 0f..1f, steps = 99) {
                        vm.setModelSettings(m, s.copy(threshold = (it * 100).roundToInt() / 100f))
                    }
                }
                SliderRow("Merge: lặng ≥", "${s.minSilenceMs} ms", s.minSilenceMs.toFloat(), 0f..2000f) {
                    vm.setModelSettings(m, s.copy(minSilenceMs = (it / 10).roundToInt() * 10))
                }
                SliderRow("Drop: speech <", "${s.minSpeechMs} ms", s.minSpeechMs.toFloat(), 0f..2000f) {
                    vm.setModelSettings(m, s.copy(minSpeechMs = (it / 10).roundToInt() * 10))
                }
                SliderRow("Pre pad", "${s.prePadMs} ms", s.prePadMs.toFloat(), 0f..500f) {
                    vm.setModelSettings(m, s.copy(prePadMs = (it / 10).roundToInt() * 10))
                }
                SliderRow("Post pad", "${s.postPadMs} ms", s.postPadMs.toFloat(), 0f..500f) {
                    vm.setModelSettings(m, s.copy(postPadMs = (it / 10).roundToInt() * 10))
                }
                if (m == VadModelType.FIRERED) {
                    SliderRow("Làm mượt", "${s.smoothWindowMs} ms", s.smoothWindowMs.toFloat(), 10f..300f) {
                        vm.setModelSettings(m, s.copy(smoothWindowMs = (it / 10).roundToInt() * 10))
                    }
                    SliderRow("Speech tối đa", String.format(Locale.US, "%.0f s", s.maxSpeechSec), s.maxSpeechSec, 1f..60f) {
                        vm.setModelSettings(m, s.copy(maxSpeechSec = it.roundToInt().toFloat()))
                    }
                }
            }
        }
    }
    SliderRow("Speech trong bin ≥", "${(st.binRatio * 100).roundToInt()}%", st.binRatio, 0.05f..1f, steps = 18) {
        vm.setBinRatio((it * 20).roundToInt() / 20f)
    }
    Hint("Bin GT ${BIN_SEC}s là speech theo model khi phần speech (sau hậu xử lý) chiếm ≥ tỷ lệ này. " +
        "Kết quả model được cache: chạy lại sau khi đổi tham số chỉ mất thời gian quét ngưỡng.")
    if (st.testOnly) {
        TextButton(onClick = vm::reloadUserSettings, enabled = !st.busy) { Text("Nạp lại cài đặt user") }
        Hint("Đổi ngưỡng không cần chạy lại: bảng và ROC cập nhật ngay. Đổi merge/drop/pad hoặc tỷ lệ bin thì cần chạy lại.")
    } else {
        SearchControls(st, vm)
    }
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = vm::run, enabled = !st.busy) { Text(if (st.evals.isEmpty()) "Chạy & đánh giá" else "Chạy lại") }
        if (st.stale) Text("Tham số/split đã đổi – cần chạy lại", color = MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun ParamsSection(st: DevState, vm: DevViewModel) = Section(if (st.testOnly) "4. Thông số báo cáo" else "4. Chọn ngưỡng") {
    val p = st.params
    if (!st.testOnly) {
        ChipRow("Tiêu chí (trên dev, strict)", Criterion.entries, p.criterion, { it.label }) { vm.setParams(p.copy(criterion = it)) }
    }
    ChipRow("Cấu hình hậu xử lý cho tầng B", PostConfig.entries, p.config, { it.code }) { vm.setParams(p.copy(config = it)) }
    Hint(p.config.description)
    SliderRow("Trọng số miss DCF", String.format(Locale.US, "%.2f", p.missWeight), p.missWeight.toFloat(), 0f..1f, steps = 19) {
        vm.setParams(p.copy(missWeight = (it * 20).roundToInt() / 20.0))
    }
    if ((p.criterion == Criterion.MR_MAX || p.criterion == Criterion.MR_FAR) && !st.testOnly) {
        SliderRow("MR tối đa", fmtPct(p.mrMax), p.mrMax.toFloat(), 0.005f..0.3f, steps = 58) {
            vm.setParams(p.copy(mrMax = (it * 200).roundToInt() / 200.0))
        }
    }
    if (p.criterion == Criterion.MR_FAR && !st.testOnly) {
        SliderRow("FAR tối đa", fmtPct(p.farMax), p.farMax.toFloat(), 0.01f..0.5f, steps = 48) {
            vm.setParams(p.copy(farMax = (it * 100).roundToInt() / 100.0))
        }
        Hint("Cả hai giới hạn phải thỏa trên dev (strict). Trong các bộ thỏa, chọn bộ có max(MR/x, FAR/y) nhỏ nhất " +
            "(biên an toàn lớn nhất). Để chừa biên cho test, đặt x, y chặt hơn mục tiêu một chút.")
    }
    ChipRow("Mốc FPR 1", listOf(0.005, 0.01, 0.02), p.fpr1, { fmtPct(it) }) { vm.setParams(p.copy(fpr1 = it)) }
    ChipRow("Mốc FPR 2", listOf(0.05, 0.1, 0.2), p.fpr2, { fmtPct(it) }) { vm.setParams(p.copy(fpr2 = it)) }
}

@Composable
private fun RocSection(vm: DevViewModel, st: DevState) = Section("5. ROC · cấu hình ${st.params.config.code}") {
    var chosenSet by rememberSaveable { mutableIntStateOf(0) }
    var mode by rememberSaveable { mutableIntStateOf(0) }
    val set = if (st.testOnly) 1 else chosenSet
    if (!st.testOnly) ChipRow("Tập", listOf(0, 1), chosenSet, { if (it == 0) "Dev" else "Test" }) { chosenSet = it }
    ChipRow("Chế độ", EvalMode.entries.indices.toList(), mode, { EvalMode.entries[it].label }) { mode = it }
    // st is read so the chart recomposes when summaries change.
    val series = remember(st.summaries, st.params, set, mode) { vm.rocSeries(set, EvalMode.entries[mode]) }
    RocChart(series, Modifier.fillMaxWidth().widthIn(max = 520.dp).aspectRatio(0.85f))
    Hint(if (st.testOnly) "Quét ngưỡng với merge/drop/pad cố định; chấm tròn = cấu hình với ngưỡng bạn chọn."
        else "Chấm tròn = điểm làm việc tại ngưỡng đã chọn.")
}

@Composable
private fun CompareSection(st: DevState) = Section("6. So sánh model") {
    val c = st.params.config.ordinal
    val series = st.summaries.map { "${it.model.label}" to modelColor(it.model) }
    val sets = if (st.testOnly) listOf(1) else listOf(0, 1)
    Text("AUC theo chế độ (${if (st.testOnly) "test" else "dev · test"}), cấu hình ${st.params.config.code}",
        style = MaterialTheme.typography.labelLarge)
    GroupedBarChart(
        groups = EvalMode.entries.flatMap { m -> sets.map { "${m.short} ${if (it == 0) "dev" else "test"}" } },
        series = series,
        values = EvalMode.entries.flatMap { m ->
            sets.map { set ->
                st.summaries.map { (if (set == 0) it.rows[c].aucDev else it.rows[c].aucTest)[m.ordinal] }
            }
        },
        maxValue = 1.0, format = { String.format(Locale.US, "%.2f", it) },
        modifier = Modifier.fillMaxWidth().height(200.dp),
    )
    Text(if (st.testOnly) "Test, strict, tại ngưỡng của từng model" else "Test, strict, tại ngưỡng đã chọn", style = MaterialTheme.typography.labelLarge)
    GroupedBarChart(
        groups = listOf("F1", "MR", "FAR", "DCF"),
        series = series,
        values = listOf(
            st.summaries.map { it.test[0].f1 },
            st.summaries.map { it.test[0].mr },
            st.summaries.map { it.test[0].far },
            st.summaries.map { it.test[0].dcf(st.params.missWeight) },
        ),
        maxValue = 1.0, format = { String.format(Locale.US, "%.2f", it) },
        modifier = Modifier.fillMaxWidth().height(200.dp),
    )
    Text("C. Bảng so sánh", style = MaterialTheme.typography.labelLarge)
    ModelComparisonTable(st.summaries, st.params)
}

@Composable
private fun SummarySection(ms: ModelSummary, st: DevState, onApply: (VadModelType, SegmentSettings) -> Unit) =
    Section("7. ${ms.model.label} · ${ms.eval.source.fileName}") {
        if (ms.eval.failed > 0) Text("${ms.eval.failed} file không phân tích được (bỏ qua)",
            color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.labelSmall)
        ModelSummaryTables(ms, st.params, st.testOnly)
        if (st.testOnly) return@Section
        val apply = ms.applicableSettings()
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { apply?.let { onApply(ms.model, it) } }, enabled = apply != null && !st.stale) {
                Text("Áp dụng cho user")
            }
            Text(
                when {
                    apply == null && ms.config == PostConfig.A0 -> "A0 không có hậu xử lý tương ứng trong app"
                    apply == null -> "Không chọn được ngưỡng"
                    else -> "ngưỡng ${String.format(Locale.US, "%.2f", apply.threshold)}, merge ${apply.minSilenceMs}, " +
                        "drop ${apply.minSpeechMs}, pad ${apply.prePadMs}/${apply.postPadMs} ms"
                },
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.weight(1f),
            )
        }
    }

@Composable
private fun SearchControls(st: DevState, vm: DevViewModel) {
    HorizontalDivider(Modifier.padding(vertical = 4.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Tìm bộ hậu xử lý tối ưu (A5)", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
        Switch(checked = st.searchEnabled, onCheckedChange = vm::setSearchEnabled, enabled = !st.busy)
    }
    if (!st.searchEnabled) {
        Hint("Tắt: A5 = A4. Bật: thử mọi tổ hợp merge × drop × pre pad × post pad × ngưỡng trên tập dev " +
            "(FireRedVAD thêm trục làm mượt), chọn tổ hợp tốt nhất theo tiêu chí, rồi quét ngưỡng đầy đủ cho A5.")
        return
    }
    val g = st.grid
    GridField("Merge: lặng ≥ (ms)", g.merge) { vm.setGrid(g.copy(merge = it)) }
    GridField("Drop: speech < (ms)", g.drop) { vm.setGrid(g.copy(drop = it)) }
    GridField("Pre pad (ms)", g.prePad) { vm.setGrid(g.copy(prePad = it)) }
    GridField("Post pad (ms)", g.postPad) { vm.setGrid(g.copy(postPad = it)) }
    val fireRed = st.models[VadModelType.FIRERED]?.let { it.enabled && it.source != null } == true
    if (fireRed) {
        GridField("Làm mượt FireRedVAD (ms)", g.smooth) { vm.setGrid(g.copy(smooth = it)) }
        Hint("Chỉ áp dụng cho FireRedVAD (bước 10 ms, < 20 ms = không làm mượt). Nhập một giá trị để cố định.")
    }
    ChipRow("Bước ngưỡng khi tìm", listOf(5, 2, 1), g.thrStep, { String.format(Locale.US, "%.2f", it / 100.0) }) {
        vm.setGrid(g.copy(thrStep = it))
    }
    val p = st.params
    ChipRow("Tìm theo tiêu chí (dev, strict)", Criterion.entries, p.criterion, { it.label }) { vm.setParams(p.copy(criterion = it)) }
    if (p.criterion == Criterion.DCF_MIN) {
        SliderRow("Trọng số miss DCF", String.format(Locale.US, "%.2f", p.missWeight), p.missWeight.toFloat(), 0f..1f, steps = 19) {
            vm.setParams(p.copy(missWeight = (it * 20).roundToInt() / 20.0))
        }
    }
    if (p.criterion == Criterion.MR_MAX || p.criterion == Criterion.MR_FAR) {
        Hint("Giới hạn MR${if (p.criterion == Criterion.MR_FAR) "/FAR" else ""} chỉnh ở mục 4 (x = ${fmtPct(p.mrMax)}" +
            (if (p.criterion == Criterion.MR_FAR) ", y = ${fmtPct(p.farMax)}" else "") + ").")
    }
    val devFiles = st.split?.devIndices()?.size ?: 0
    val models = st.models.filter { it.value.enabled && it.value.source != null }.keys
    val combos = models.sumOf { g.size(it == VadModelType.FIRERED).toLong() }
    val calls = combos * g.thresholds.size * devFiles
    Text(
        "$combos tổ hợp (${models.size} model) × ${g.thresholds.size} ngưỡng × $devFiles file dev = " +
            String.format(Locale.US, "%,d", calls) + " lần tách đoạn",
        style = MaterialTheme.typography.labelSmall,
        color = if (calls > 20_000_000) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

/** Comma-separated list of integers; keeps the raw text while typing, applies it when it parses. */
@Composable
private fun GridField(label: String, values: List<Int>, onChange: (List<Int>) -> Unit) {
    var text by rememberSaveable(label) { mutableStateOf(SearchGrid.format(values)) }
    val parsed = SearchGrid.parse(text)
    OutlinedTextField(
        value = text,
        onValueChange = {
            text = it
            val v = SearchGrid.parse(it)
            if (v.isNotEmpty()) onChange(v)
        },
        label = { Text("$label · ${parsed.size} giá trị") },
        isError = parsed.isEmpty(),
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun SearchSection(ms: ModelSummary, st: DevState, vm: DevViewModel) {
    val r = ms.eval.search ?: return
    Section("Tìm tham số · ${ms.model.label}") {
        val best = r.best
        val name = r.criterion.label + when (r.criterion) {
            Criterion.DCF_MIN -> " (w = ${r.missWeight})"
            Criterion.MR_MAX -> " (x = ${fmtPct(r.mrMax)})"
            Criterion.MR_FAR -> " (x = ${fmtPct(r.mrMax)}, y = ${fmtPct(r.farMax)})"
            else -> ""
        }
        Text(
            "${r.entries.size} tổ hợp × ${r.grid.thresholds.size} ngưỡng trên ${r.devFiles} file dev · " +
                "${r.elapsedMs / 1000} s · tiêu chí $name",
            style = MaterialTheme.typography.labelSmall,
        )
        if (vm.searchOutdated(st)) {
            Text("Tiêu chí/trọng số đã đổi sau khi tìm – A5 được tìm theo $name, chạy lại để tìm lại.",
                color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.labelSmall)
        }
        if (best == null) {
            Text("Không tổ hợp nào thỏa tiêu chí.", color = MaterialTheme.colorScheme.error)
            return@Section
        }
        val b = best.settings
        Text(
            "Tốt nhất: merge ${b.minSilenceMs} · drop ${b.minSpeechMs} · pad ${b.prePadMs}/${b.postPadMs} ms · " +
                (if (r.withSmooth) "làm mượt ${b.smoothWindowMs} ms · " else "") +
                "ngưỡng ${fmtThr(best.bestK)} → ${fmt3(r.value(best))}",
            style = MaterialTheme.typography.bodyMedium,
        )
        // Same criterion on the full threshold grid, A4 (user settings) vs A5 (tuned), dev strict.
        val a4 = ms.rows[PostConfig.A4.ordinal]
        val a5 = ms.rows[PostConfig.A5.ordinal]
        val p = st.params
        fun objective(row: Int) = Metrics.selectBy(
            NUM_THR, { Metrics.conf(ms.eval.devSweep, row, it, 0) }, r.criterion, r.missWeight, r.mrMax, r.farMax,
        ).second.let { Metrics.costToValue(r.criterion, it) }
        Text(
            "Dev, lưới ngưỡng 0.01: A4 ${fmt3(objective(a4.config.ordinal))} → A5 ${fmt3(objective(a5.config.ordinal))} · " +
                "AUC dev A4 ${fmt3(a4.aucDev[0])} / A5 ${fmt3(a5.aucDev[0])}",
            style = MaterialTheme.typography.labelSmall,
        )
        Text("Top 10 tổ hợp (dev, strict)", style = MaterialTheme.typography.labelLarge)
        DataTable(
            header = listOf("#", "Merge", "Drop", "Pre", "Post") + (if (r.withSmooth) listOf("Mượt") else emptyList()) +
                listOf("Ngưỡng", r.criterion.excel, "F1", "MR", "FAR", "DCF"),
            rows = r.ranked().take(10).mapIndexed { i, e ->
                val c = e.conf[0]
                val s = e.settings
                listOf("${i + 1}", "${s.minSilenceMs}", "${s.minSpeechMs}", "${s.prePadMs}", "${s.postPadMs}") +
                    (if (r.withSmooth) listOf("${s.smoothWindowMs}") else emptyList()) +
                    listOf(fmtThr(e.bestK), fmt3(r.value(e)), fmt3(c.f1), fmtPct(c.mr), fmtPct(c.far), fmt3(c.dcf(p.missWeight)))
            },
            firstColWidth = 32.dp,
            colWidth = 64.dp,
            highlightRow = 0,
        )
        Text("Ảnh hưởng từng tham số (tốt nhất theo các tham số còn lại)", style = MaterialTheme.typography.labelLarge)
        val values = r.params.map { r.marginal(it) }
        val finite = values.flatten().map { it.second }.filter { !it.isNaN() }
        val maxV = (finite.maxOrNull() ?: 1.0).let { if (it <= 0) 1.0 else it * 1.1 }
        for (param in r.params) {
            val m = values[param]
            GroupedBarChart(
                groups = m.map { "${it.first}" },
                series = listOf(SearchResult.PARAM_NAMES[param] to modelColor(ms.model)),
                values = m.map { listOf(it.second) },
                maxValue = maxV, format = { String.format(Locale.US, "%.3f", it) },
                modifier = Modifier.fillMaxWidth().height(130.dp),
            )
        }
    }
}
