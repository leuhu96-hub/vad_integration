package com.example.vadapp

import android.app.Application
import android.content.res.AssetFileDescriptor
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.NativeVad
import com.example.vadapp.vad.Segment
import com.example.vadapp.vad.SegmentSettings
import com.example.vadapp.vad.VadModelType
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class UiState(
    val model: VadModelType = VadModelType.FIRERED,
    val settings: Map<VadModelType, SegmentSettings> = VadModelType.entries.associateWith { it.defaults },
    val videoUri: Uri? = null,
    val videoName: String? = null,
    val analyzing: Boolean = false,
    val progress: Float = 0f,
    val result: AnalysisResult? = null,
    val segments: List<Segment> = emptyList(),
    val error: String? = null,
) {
    val currentSettings: SegmentSettings get() = settings.getValue(model)
}

data class PlaybackState(
    val positionMs: Long = 0,
    val durationMs: Long = 0,
    val isPlaying: Boolean = false,
    /** Segment being played in "play this segment only" mode, auto-pauses at its end. */
    val playingSegment: Int = -1,
    /** Segment under the playhead, or -1. */
    val currentSegment: Int = -1,
    val speechOnly: Boolean = false,
    val videoAspect: Float = 16f / 9f,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {

    val player: ExoPlayer = ExoPlayer.Builder(app).build()

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    private val _playback = MutableStateFlow(PlaybackState())
    val playback: StateFlow<PlaybackState> = _playback.asStateFlow()

    // Results are cached per (video, model) so switching models back and forth is instant.
    private val cache = HashMap<Pair<Uri, VadModelType>, AnalysisResult>()
    private var analysisJob: Job? = null
    private var stopAtMs: Long = -1

    init {
        player.addListener(object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: androidx.media3.common.VideoSize) {
                if (videoSize.width > 0 && videoSize.height > 0) {
                    val aspect = videoSize.width * videoSize.pixelWidthHeightRatio / videoSize.height
                    _playback.update { it.copy(videoAspect = aspect) }
                }
            }
        })
        viewModelScope.launch { pollPlayer() }
    }

    // ------------------------------------------------------------------ input

    fun selectModel(model: VadModelType) {
        if (model == _ui.value.model) return
        _ui.update { it.copy(model = model) }
        analyze()
    }

    fun selectVideo(uri: Uri) {
        val name = queryName(uri)
        cache.keys.removeAll { it.first != uri }
        _ui.update { it.copy(videoUri = uri, videoName = name, result = null, segments = emptyList(), error = null) }
        stopAtMs = -1
        _playback.update { PlaybackState(speechOnly = it.speechOnly) }
        player.setMediaItem(MediaItem.fromUri(uri))
        player.prepare()
        analyze()
    }

    fun updateSettings(settings: SegmentSettings) {
        _ui.update { it.copy(settings = it.settings + (it.model to settings)) }
        resegment()
    }

    fun resetSettings() = updateSettings(_ui.value.model.defaults)

    fun cancelAnalysis() {
        analysisJob?.cancel()
    }

    private fun analyze() {
        val state = _ui.value
        val uri = state.videoUri ?: return
        val model = state.model
        analysisJob?.cancel()

        cache[uri to model]?.let { cached ->
            _ui.update { it.copy(analyzing = false, result = cached, error = null) }
            resegment()
            return
        }

        _ui.update { it.copy(analyzing = true, progress = 0f, result = null, segments = emptyList(), error = null) }
        analysisJob = viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    val job = coroutineContext[Job]
                    runNative(uri, model) { job?.isActive != false }
                }
                cache[uri to model] = result
                if (_ui.value.model == model && _ui.value.videoUri == uri) {
                    _ui.update { it.copy(analyzing = false, result = result) }
                    resegment()
                }
            } catch (e: CancellationException) {
                if (_ui.value.model == model && _ui.value.videoUri == uri) {
                    _ui.update { it.copy(analyzing = false) }
                }
                throw e
            } catch (e: Exception) {
                _ui.update { it.copy(analyzing = false, error = e.message ?: e.toString()) }
            }
        }
    }

    private fun runNative(uri: Uri, model: VadModelType, keepGoing: () -> Boolean): AnalysisResult {
        val app = getApplication<Application>()
        val afd = app.contentResolver.openAssetFileDescriptor(uri, "r")
            ?: error("Không mở được file")
        return afd.use {
            val length = if (it.length == AssetFileDescriptor.UNKNOWN_LENGTH) Long.MAX_VALUE else it.length
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            NativeVad.analyze(app.assets, model.id, it.parcelFileDescriptor.fd, it.startOffset, length, threads) { p ->
                _ui.update { s -> s.copy(progress = p) }
                keepGoing()
            }
        }
    }

    private fun resegment() {
        val state = _ui.value
        val result = state.result ?: return
        val segments = NativeVad.segments(state.model, result, state.currentSettings)
        _ui.update { it.copy(segments = segments) }
        stopAtMs = -1
        _playback.update { it.copy(playingSegment = -1) }
    }

    private fun queryName(uri: Uri): String? = runCatching {
        getApplication<Application>().contentResolver
            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
    }.getOrNull() ?: uri.lastPathSegment

    // ------------------------------------------------------------------ playback

    fun togglePlay() {
        if (player.isPlaying) {
            player.pause()
        } else {
            if (player.playbackState == Player.STATE_ENDED) player.seekTo(0)
            player.play()
        }
    }

    /** Free seek (timeline / scrubbing): leaves "segment only" playback. */
    fun seekTo(ms: Long) {
        stopAtMs = -1
        _playback.update { it.copy(playingSegment = -1) }
        player.seekTo(ms.coerceAtLeast(0))
    }

    /** Plays exactly one segment and pauses at its end. */
    fun playSegment(index: Int) {
        val seg = _ui.value.segments.getOrNull(index) ?: return
        stopAtMs = seg.endMs
        _playback.update { it.copy(playingSegment = index) }
        player.seekTo(seg.startMs)
        player.play()
    }

    fun nextSegment() {
        val segs = _ui.value.segments
        val pos = player.currentPosition
        val idx = segs.indexOfFirst { it.startMs > pos + 50 }
        if (idx >= 0) jumpTo(idx)
    }

    fun previousSegment() {
        val segs = _ui.value.segments
        val pos = player.currentPosition
        // Like a music player: restart the current segment unless we're right at its start.
        val idx = segs.indexOfLast { it.startMs < pos - 700 }
        if (idx >= 0) jumpTo(idx)
    }

    private fun jumpTo(index: Int) {
        val playing = player.isPlaying || _playback.value.playingSegment >= 0
        if (_playback.value.playingSegment >= 0) {
            playSegment(index)
        } else {
            seekTo(_ui.value.segments[index].startMs)
            if (playing) player.play()
        }
    }

    fun setSpeechOnly(enabled: Boolean) {
        _playback.update { it.copy(speechOnly = enabled) }
    }

    private suspend fun pollPlayer() {
        while (currentCoroutineContext().isActive) {
            val pos = player.currentPosition
            val segs = _ui.value.segments
            var pb = _playback.value

            if (player.isPlaying) {
                if (stopAtMs >= 0 && pos >= stopAtMs) {
                    player.pause()
                    player.seekTo(stopAtMs)
                    stopAtMs = -1
                    pb = pb.copy(playingSegment = -1)
                } else if (pb.speechOnly && stopAtMs < 0 && segs.isNotEmpty()) {
                    // Skip silence: jump over gaps to the next speech segment.
                    val inside = segs.any { pos >= it.startMs && pos < it.endMs }
                    if (!inside) {
                        val next = segs.firstOrNull { it.startMs > pos }
                        if (next != null) player.seekTo(next.startMs) else player.pause()
                    }
                }
            }

            val current = segs.indexOfFirst { pos >= it.startMs && pos < it.endMs }
            val duration = player.duration.takeIf { it > 0 }
                ?: _ui.value.result?.let { (it.durationSec * 1000).toLong() } ?: 0L
            _playback.value = pb.copy(
                positionMs = pos,
                durationMs = duration,
                isPlaying = player.isPlaying,
                currentSegment = current,
            )
            delay(if (player.isPlaying) 30 else 120)
        }
    }

    override fun onCleared() {
        analysisJob?.cancel()
        player.release()
    }
}
