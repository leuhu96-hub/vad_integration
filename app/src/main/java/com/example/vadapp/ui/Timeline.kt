package com.example.vadapp.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.Segment
import kotlin.math.max
import kotlin.math.min

/**
 * Draws the speech-probability curve, the threshold and the speech segments for the time
 * window [viewStartSec, viewEndSec], plus the playhead. Tap or drag to seek.
 */
@Composable
fun Timeline(
    result: AnalysisResult?,
    segments: List<Segment>,
    threshold: Float,
    viewStartSec: Float,
    viewEndSec: Float,
    positionSec: Float,
    highlightSegment: Int,
    onSeek: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    val segColor = colors.primary
    val curveColor = colors.tertiary
    val bg = colors.surfaceVariant
    val gridColor = colors.outlineVariant
    val playheadColor = colors.error

    val range = max(1e-3f, viewEndSec - viewStartSec)
    val start by rememberUpdatedState(viewStartSec)
    val span by rememberUpdatedState(range)
    val seek by rememberUpdatedState(onSeek)

    Canvas(
        modifier
            .pointerInput(Unit) {
                detectTapGestures { o -> seek(start + o.x / size.width * span) }
            }
            .pointerInput(Unit) {
                detectHorizontalDragGestures { change, _ ->
                    change.consume()
                    seek(start + (change.position.x / size.width).coerceIn(0f, 1f) * span)
                }
            }
    ) {
        val w = size.width
        val h = size.height
        fun x(t: Float) = (t - viewStartSec) / range * w

        drawRoundRect(bg, cornerRadius = CornerRadius(12f, 12f))

        // Speech segments: translucent band + solid strip at the bottom.
        segments.forEachIndexed { i, s ->
            if (s.endSec < viewStartSec || s.startSec > viewEndSec) return@forEachIndexed
            val x0 = max(0f, x(s.startSec))
            val x1 = min(w, x(s.endSec))
            val alpha = if (i == highlightSegment) 0.40f else 0.18f
            drawRect(segColor.copy(alpha = alpha), Offset(x0, 0f), Size(max(1f, x1 - x0), h))
            drawRect(segColor, Offset(x0, h - 8f), Size(max(1f, x1 - x0), 8f))
        }

        // Probability curve, one min/max column per pixel so long files stay cheap to draw.
        if (result != null && result.probs.isNotEmpty()) {
            val probs = result.probs
            val path = Path()
            val plotH = h - 12f
            val cols = w.toInt().coerceAtLeast(1)
            var started = false
            for (px in 0 until cols) {
                val t0 = viewStartSec + px / w * range
                val t1 = viewStartSec + (px + 1) / w * range
                val i0 = ((t0 - result.offsetSec) / result.hopSec).toInt().coerceIn(0, probs.size - 1)
                val i1 = ((t1 - result.offsetSec) / result.hopSec).toInt().coerceIn(i0, probs.size - 1)
                var hi = 0f
                for (i in i0..i1) hi = max(hi, probs[i])
                val y = plotH - hi * (plotH - 4f)
                if (!started) {
                    path.moveTo(px.toFloat(), y); started = true
                } else {
                    path.lineTo(px.toFloat(), y)
                }
            }
            drawPath(path, curveColor, style = Stroke(width = 2.5f))

            val ty = plotH - threshold * (plotH - 4f)
            drawLine(
                gridColor, Offset(0f, ty), Offset(w, ty), strokeWidth = 2f,
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f)),
            )
        }

        if (positionSec in viewStartSec..viewEndSec) {
            val px = x(positionSec)
            drawLine(playheadColor, Offset(px, 0f), Offset(px, h), strokeWidth = 4f)
        }
    }
}
