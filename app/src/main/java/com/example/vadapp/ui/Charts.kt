package com.example.vadapp.ui

import android.graphics.Bitmap
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import com.example.vadapp.vad.VadModelType
import kotlin.math.max

/** Fixed, distinguishable series colours (also used in the exported PNGs). */
fun modelColor(model: VadModelType): Color = when (model) {
    VadModelType.FIRERED -> Color(0xFF2A6FDB)
    VadModelType.SILERO -> Color(0xFFE8710A)
    VadModelType.YAMNET -> Color(0xFF1E8E3E)
}

class RocSeries(
    val label: String,
    val color: Color,
    /** (FAR, TPR) points in any order. */
    val points: List<Pair<Double, Double>>,
    /** Operating point at the chosen threshold. */
    val marker: Pair<Double, Double>?,
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun Legend(entries: List<Pair<String, Color>>, modifier: Modifier = Modifier) {
    FlowRow(modifier, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        entries.forEach { (label, color) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(10.dp).background(color))
                Spacer(Modifier.width(4.dp))
                Text(label, style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}

/** ROC plot: FAR on x, TPR (recall) on y, diagonal for reference, a dot at the chosen threshold. */
@Composable
fun RocChart(series: List<RocSeries>, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme
    val axis = colors.onSurfaceVariant
    val grid = colors.outlineVariant
    val textColor = axis.toArgb()
    Column(modifier) {
        Canvas(Modifier.weight(1f).fillMaxWidth()) {
            val padL = 64f
            val padB = 48f
            val padT = 12f
            val padR = 12f
            val w = size.width - padL - padR
            val h = size.height - padT - padB
            fun px(x: Double) = padL + (x * w).toFloat()
            fun py(y: Double) = padT + ((1 - y) * h).toFloat()

            val paint = Paint().apply { color = textColor; textSize = 26f; isAntiAlias = true }
            for (i in 0..5) {
                val v = i / 5.0
                drawLine(grid, Offset(px(v), py(0.0)), Offset(px(v), py(1.0)), strokeWidth = 1f)
                drawLine(grid, Offset(px(0.0), py(v)), Offset(px(1.0), py(v)), strokeWidth = 1f)
                drawContext.canvas.nativeCanvas.apply {
                    paint.textAlign = Paint.Align.CENTER
                    drawText("%.1f".format(v), px(v), size.height - padB + 30f, paint)
                    paint.textAlign = Paint.Align.RIGHT
                    drawText("%.1f".format(v), padL - 8f, py(v) + 9f, paint)
                }
            }
            drawLine(
                axis, Offset(px(0.0), py(0.0)), Offset(px(1.0), py(1.0)), strokeWidth = 1.5f,
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f)),
            )
            for (s in series) {
                val pts = (s.points + listOf(0.0 to 0.0, 1.0 to 1.0)).sortedWith(compareBy({ it.first }, { it.second }))
                val path = Path()
                pts.forEachIndexed { i, (x, y) ->
                    if (i == 0) path.moveTo(px(x), py(y)) else path.lineTo(px(x), py(y))
                }
                drawPath(path, s.color, style = Stroke(width = 4f))
                s.marker?.let { (x, y) ->
                    drawCircle(Color.White, 9f, Offset(px(x), py(y)))
                    drawCircle(s.color, 7f, Offset(px(x), py(y)))
                }
            }
        }
        Text("x: FAR (1 − Specificity) · y: Recall (1 − MR)", style = MaterialTheme.typography.labelSmall, color = axis)
        Legend(series.map { it.label to it.color })
    }
}

/**
 * Grouped bar chart: one group per [groups] entry, one bar per series. Values are in 0..[maxValue];
 * NaN values are skipped.
 */
@Composable
fun GroupedBarChart(
    groups: List<String>,
    series: List<Pair<String, Color>>,
    values: List<List<Double>>,
    maxValue: Double,
    format: (Double) -> String,
    modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    val textColor = colors.onSurfaceVariant.toArgb()
    val grid = colors.outlineVariant
    Column(modifier) {
        Canvas(Modifier.weight(1f).fillMaxWidth()) {
            val padB = 40f
            val padT = 28f
            val h = size.height - padB - padT
            val groupW = size.width / max(1, groups.size)
            val barW = groupW * 0.8f / max(1, series.size)
            val paint = Paint().apply { color = textColor; textSize = 24f; isAntiAlias = true; textAlign = Paint.Align.CENTER }
            drawLine(grid, Offset(0f, padT + h), Offset(size.width, padT + h), strokeWidth = 1.5f)
            groups.forEachIndexed { g, label ->
                val gx = g * groupW + groupW * 0.1f
                series.forEachIndexed { si, (_, color) ->
                    val v = values.getOrNull(g)?.getOrNull(si) ?: Double.NaN
                    if (v.isNaN()) return@forEachIndexed
                    val bh = (v.coerceIn(0.0, maxValue) / maxValue * h).toFloat()
                    val x = gx + si * barW
                    drawRect(color, Offset(x + 2f, padT + h - bh), Size(barW - 4f, bh))
                    drawContext.canvas.nativeCanvas.drawText(format(v), x + barW / 2, padT + h - bh - 6f, paint)
                }
                drawContext.canvas.nativeCanvas.drawText(label, g * groupW + groupW / 2, size.height - 10f, paint)
            }
        }
        Legend(series)
    }
}

/** Renders a ROC plot to a bitmap for export (white background, same colours as the app). */
fun renderRocBitmap(title: String, series: List<RocSeries>, sizePx: Int = 1200): Bitmap {
    val bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888)
    val c = android.graphics.Canvas(bmp)
    c.drawColor(android.graphics.Color.WHITE)
    val padL = 110f
    val padB = 110f
    val padT = 90f
    val padR = 40f
    val legendH = 50f * series.size
    val w = sizePx - padL - padR
    val h = sizePx - padT - padB - legendH
    fun px(x: Double) = padL + (x * w).toFloat()
    fun py(y: Double) = padT + ((1 - y) * h).toFloat()
    val text = Paint().apply { color = android.graphics.Color.DKGRAY; textSize = 30f; isAntiAlias = true }
    val gridP = Paint().apply { color = 0xFFDDDDDD.toInt(); strokeWidth = 2f }
    text.textSize = 40f
    c.drawText(title, padL, 60f, text)
    text.textSize = 28f
    for (i in 0..10) {
        val v = i / 10.0
        c.drawLine(px(v), py(0.0), px(v), py(1.0), gridP)
        c.drawLine(px(0.0), py(v), px(1.0), py(v), gridP)
        text.textAlign = Paint.Align.CENTER
        c.drawText("%.1f".format(v), px(v), py(0.0) + 40f, text)
        text.textAlign = Paint.Align.RIGHT
        c.drawText("%.1f".format(v), padL - 12f, py(v) + 10f, text)
    }
    text.textAlign = Paint.Align.CENTER
    c.drawText("FAR (1 − Specificity)", padL + w / 2, py(0.0) + 85f, text)
    c.save()
    c.rotate(-90f, 35f, padT + h / 2)
    c.drawText("Recall (1 − MR)", 35f, padT + h / 2, text)
    c.restore()
    val diag = Paint().apply {
        color = android.graphics.Color.GRAY; strokeWidth = 2f; style = Paint.Style.STROKE
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(12f, 12f), 0f)
    }
    c.drawLine(px(0.0), py(0.0), px(1.0), py(1.0), diag)
    series.forEachIndexed { i, s ->
        val p = Paint().apply { color = s.color.toArgb(); strokeWidth = 5f; style = Paint.Style.STROKE; isAntiAlias = true }
        val pts = (s.points + listOf(0.0 to 0.0, 1.0 to 1.0)).sortedWith(compareBy({ it.first }, { it.second }))
        val path = android.graphics.Path()
        pts.forEachIndexed { j, (x, y) -> if (j == 0) path.moveTo(px(x), py(y)) else path.lineTo(px(x), py(y)) }
        c.drawPath(path, p)
        s.marker?.let { (x, y) ->
            p.style = Paint.Style.FILL
            c.drawCircle(px(x), py(y), 11f, p)
        }
        val ly = sizePx - legendH + i * 50f + 10f
        val fill = Paint().apply { color = s.color.toArgb() }
        c.drawRect(padL, ly, padL + 30f, ly + 30f, fill)
        text.textAlign = Paint.Align.LEFT
        c.drawText(s.label, padL + 45f, ly + 27f, text)
    }
    return bmp
}
