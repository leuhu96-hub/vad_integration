package com.example.vadapp.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeveloperMode
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SkipNext
import androidx.compose.material.icons.filled.SkipPrevious
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.media3.common.util.UnstableApi
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import com.example.vadapp.MainViewModel
import com.example.vadapp.PlaybackState
import com.example.vadapp.UiState
import com.example.vadapp.vad.ModelSource
import com.example.vadapp.vad.Segment
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.VadModelType
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val DETAIL_WINDOW_SEC = 20f

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(vm: MainViewModel, onOpenDev: () -> Unit) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    val pb by vm.playback.collectAsStateWithLifecycle()
    var showSettings by rememberSaveable { mutableStateOf(false) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.selectVideo(uri)
    }
    val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.importModel(uri)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("VAD Video", style = MaterialTheme.typography.titleMedium)
                        Text(
                            ui.videoName ?: "Chưa chọn file",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                },
                actions = {
                    ModelMenu(
                        model = ui.model,
                        sources = ui.currentSources,
                        selected = ui.currentSource,
                        onSelectModel = vm::selectModel,
                        onSelectSource = vm::selectSource,
                        onRemove = vm::removeModel,
                        // Model files have no reliable MIME type, so let the user pick anything;
                        // the format is detected from the content.
                        onImport = { modelPicker.launch(arrayOf("*/*")) },
                    )
                    IconButton(onClick = onOpenDev) {
                        Icon(Icons.Filled.DeveloperMode, contentDescription = "Chế độ Dev")
                    }
                    IconButton(onClick = { showSettings = true }) {
                        Icon(Icons.Filled.Tune, contentDescription = "Tham số")
                    }
                    IconButton(onClick = { picker.launch(arrayOf("video/*", "audio/*")) }) {
                        Icon(Icons.Filled.VideoLibrary, contentDescription = "Chọn video")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
        ) {
            if (ui.videoUri != null) {
                VideoSurface(vm, pb.videoAspect)
                PlayerSection(ui, pb, vm)
            }

            LazyColumnBody(ui, pb, vm, onPickVideo = { picker.launch(arrayOf("video/*", "audio/*")) })
        }
    }

    if (showSettings) {
        ModalBottomSheet(
            onDismissRequest = { showSettings = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        ) {
            SettingsPanel(
                ui.model, ui.currentSettings, vm::updateSettings, vm::resetSettings,
                Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 24.dp),
            )
        }
    }
}

@Composable
private fun ModelMenu(
    model: VadModelType,
    sources: List<ModelSource>,
    selected: ModelSource?,
    onSelectModel: (VadModelType) -> Unit,
    onSelectSource: (ModelSource) -> Unit,
    onRemove: (ModelSource) -> Unit,
    onImport: () -> Unit,
) {
    var open by remember { mutableStateOf(false) }
    Box {
        FilledTonalButton(onClick = { open = true }, contentPadding = PaddingValues(start = 14.dp, end = 6.dp)) {
            Text(model.label)
            Icon(Icons.Filled.ArrowDropDown, contentDescription = "Chọn model")
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            MenuHeader("Model")
            VadModelType.entries.forEach { m ->
                DropdownMenuItem(
                    text = { Text(m.label) },
                    leadingIcon = { CheckSlot(m == model) },
                    onClick = { onSelectModel(m) },
                )
            }
            Text(
                model.description,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .widthIn(max = 260.dp)
                    .padding(horizontal = 16.dp, vertical = 4.dp),
            )
            HorizontalDivider(Modifier.padding(vertical = 4.dp))
            MenuHeader("File model · ${model.label}")
            if (sources.isEmpty()) {
                Text(
                    "chưa có file",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                )
            }
            sources.forEach { src ->
                DropdownMenuItem(
                    text = {
                        Text(
                            if (src.custom) "${src.fileName} · ${src.format.label}" else src.label,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                    leadingIcon = { CheckSlot(src == selected) },
                    trailingIcon = if (src.custom) {
                        {
                            IconButton(onClick = { onRemove(src) }) {
                                Icon(Icons.Filled.Close, contentDescription = "Xóa ${src.fileName}")
                            }
                        }
                    } else null,
                    onClick = { onSelectSource(src); open = false },
                )
            }
            DropdownMenuItem(
                text = { Text("Nạp .onnx / .tflite…") },
                leadingIcon = { Icon(Icons.Filled.FileOpen, contentDescription = null) },
                onClick = { open = false; onImport() },
            )
        }
    }
}

@Composable
private fun MenuHeader(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
    )
}

@Composable
private fun CheckSlot(checked: Boolean) {
    if (checked) Icon(Icons.Filled.Check, contentDescription = "Đang chọn")
    else Spacer(Modifier.size(24.dp))
}

@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun VideoSurface(vm: MainViewModel, aspect: Float) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = vm.player
                    useController = false
                    resizeMode = AspectRatioFrameLayout.RESIZE_MODE_FIT
                }
            },
            modifier = Modifier
                .widthIn(max = 720.dp)
                .fillMaxWidth()
                // Keep tall videos from pushing the rest of the UI off screen.
                .aspectRatio(max(aspect, 4f / 3f))
                .clickable { vm.togglePlay() },
        )
    }
}

@Composable
private fun PlayerSection(ui: UiState, pb: PlaybackState, vm: MainViewModel) {
    val durationSec = max(pb.durationMs, 1L) / 1000f
    val positionSec = pb.positionMs / 1000f
    Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = vm::previousSegment, enabled = ui.segments.isNotEmpty()) {
                Icon(Icons.Filled.SkipPrevious, contentDescription = "Đoạn trước")
            }
            FilledIconButton(onClick = vm::togglePlay) {
                Icon(
                    if (pb.isPlaying) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                    contentDescription = if (pb.isPlaying) "Tạm dừng" else "Phát",
                )
            }
            IconButton(onClick = vm::nextSegment, enabled = ui.segments.isNotEmpty()) {
                Icon(Icons.Filled.SkipNext, contentDescription = "Đoạn sau")
            }
            Text(
                "${formatTime(positionSec)} / ${formatTime(durationSec)}",
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.weight(1f),
            )
            Text("Chỉ speech", style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.width(6.dp))
            Switch(checked = pb.speechOnly, onCheckedChange = vm::setSpeechOnly)
        }

        // Detail view that follows the playhead, for precise seeking around boundaries.
        val window = min(DETAIL_WINDOW_SEC, durationSec)
        val viewStart = (positionSec - window / 2).coerceIn(0f, max(0f, durationSec - window))
        Timeline(
            result = ui.result,
            segments = ui.segments,
            threshold = ui.currentSettings.threshold,
            viewStartSec = viewStart,
            viewEndSec = viewStart + window,
            positionSec = positionSec,
            highlightSegment = pb.currentSegment,
            onSeek = { vm.seekTo((it * 1000).toLong()) },
            modifier = Modifier
                .fillMaxWidth()
                .height(72.dp),
        )
        Row(Modifier.fillMaxWidth()) {
            Text(formatTime(viewStart), style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.weight(1f))
            Text(formatTime(viewStart + window), style = MaterialTheme.typography.labelSmall)
        }

        if (ui.analyzing) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                LinearProgressIndicator(progress = { ui.progress }, modifier = Modifier.weight(1f))
                Text(
                    " ${(ui.progress * 100).roundToInt()}%",
                    style = MaterialTheme.typography.labelMedium,
                )
                TextButton(onClick = vm::cancelAnalysis) { Text("Hủy") }
            }
        }
        ui.error?.let {
            Text("Lỗi: $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
        ui.result?.let { r ->
            val speech = ui.segments.sumOf { it.durationSec.toDouble() }
            val ratio = if (r.durationSec > 0) speech / r.durationSec * 100 else 0.0
            Text(
                String.format(
                    Locale.US, "%s · %d Hz · %d kênh · %s · %d đoạn · speech %.0f%% · xử lý %.1fs",
                    r.format, r.sourceSampleRate, r.sourceChannels, r.mime.removePrefix("audio/"),
                    ui.segments.size, ratio, r.elapsedMs / 1000f,
                ),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun androidx.compose.foundation.layout.ColumnScope.LazyColumnBody(
    ui: UiState, pb: PlaybackState, vm: MainViewModel, onPickVideo: () -> Unit,
) {
    val listState = rememberLazyListState()

    // Keep the segment under the playhead visible while playing.
    LaunchedEffect(pb.currentSegment, pb.isPlaying) {
        if (pb.isPlaying && pb.currentSegment >= 0) {
            val target = pb.currentSegment
            val visible = listState.layoutInfo.visibleItemsInfo.map { it.index }
            if (target !in visible) listState.animateScrollToItem(target)
        }
    }

    LazyColumn(
        state = listState,
        modifier = Modifier
            .weight(1f)
            .fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (ui.videoUri == null) {
            item(key = "hint") {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(vertical = 32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    Text(
                        "Chọn model ở thanh trên, rồi chọn một video (hoặc file audio). App sẽ tách audio, " +
                            "chuyển về 16 kHz mono và tìm các đoạn có tiếng nói.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Button(onClick = onPickVideo) {
                        Icon(Icons.Filled.VideoLibrary, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Chọn video")
                    }
                }
            }
        }
        itemsIndexed(ui.segments, key = { i, s -> "$i-${s.startSec}" }) { i, seg ->
            SegmentRow(
                index = i,
                segment = seg,
                isCurrent = i == pb.currentSegment,
                isPlayingThis = i == pb.playingSegment && pb.isPlaying,
                onPlay = { vm.playSegment(i) },
                onSeek = { vm.seekTo(seg.startMs) },
            )
        }
    }
}

@Composable
private fun SegmentRow(
    index: Int,
    segment: Segment,
    isCurrent: Boolean,
    isPlayingThis: Boolean,
    onPlay: () -> Unit,
    onSeek: () -> Unit,
) {
    val container = if (isCurrent) MaterialTheme.colorScheme.primaryContainer
    else MaterialTheme.colorScheme.surfaceContainerLow
    Card(
        onClick = onSeek,
        colors = CardDefaults.cardColors(containerColor = container),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            Modifier.padding(start = 12.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "#${index + 1}",
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.width(48.dp),
            )
            Column(Modifier.weight(1f)) {
                Text(
                    "${formatTime(segment.startSec)} → ${formatTime(segment.endSec)}",
                    fontFamily = FontFamily.Monospace,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    String.format(Locale.US, "%.2f s", segment.durationSec),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            IconButton(onClick = onPlay) {
                Icon(
                    if (isPlayingThis) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                    contentDescription = "Phát đoạn ${index + 1}",
                )
            }
        }
    }
}

@Composable
private fun SettingsPanel(
    model: VadModelType,
    settings: SegmentSettings,
    onChange: (SegmentSettings) -> Unit,
    onReset: () -> Unit,
    modifier: Modifier = Modifier,
) {
    run {
        Column(modifier) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Tham số tách đoạn · ${model.label}",
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onReset) { Text("Mặc định") }
            }
            LabeledSlider(
                "Ngưỡng", String.format(Locale.US, "%.2f", settings.threshold),
                settings.threshold, 0.05f..0.95f,
            ) { onChange(settings.copy(threshold = it)) }
            LabeledSlider(
                "Speech tối thiểu", "${settings.minSpeechMs} ms",
                settings.minSpeechMs.toFloat(), 0f..2000f,
            ) { onChange(settings.copy(minSpeechMs = (it / 10).roundToInt() * 10)) }
            LabeledSlider(
                "Lặng tối thiểu", "${settings.minSilenceMs} ms",
                settings.minSilenceMs.toFloat(), 0f..2000f,
            ) { onChange(settings.copy(minSilenceMs = (it / 10).roundToInt() * 10)) }
            LabeledSlider(
                "Pre padding", "${settings.prePadMs} ms",
                settings.prePadMs.toFloat(), 0f..500f,
            ) { onChange(settings.copy(prePadMs = (it / 10).roundToInt() * 10)) }
            LabeledSlider(
                "Post padding", "${settings.postPadMs} ms",
                settings.postPadMs.toFloat(), 0f..500f,
            ) { onChange(settings.copy(postPadMs = (it / 10).roundToInt() * 10)) }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
) {
    // One compact row: label | slider | value.
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.width(96.dp))
        Slider(value = value, onValueChange = onChange, valueRange = range, modifier = Modifier.weight(1f))
        Text(
            valueText,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            textAlign = TextAlign.End,
            modifier = Modifier.width(60.dp),
        )
    }
}

private fun formatTime(sec: Float): String {
    val totalMs = (max(0f, sec) * 1000).roundToInt()
    val m = totalMs / 60000
    val s = (totalMs / 1000) % 60
    val ms = totalMs % 1000
    return String.format(Locale.US, "%02d:%02d.%03d", m, s, ms)
}
