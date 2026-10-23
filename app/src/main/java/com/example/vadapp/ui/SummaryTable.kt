package com.example.vadapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.vadapp.vad.eval.EvalMode
import com.example.vadapp.vad.eval.ModelSummary
import com.example.vadapp.vad.eval.PostConfig
import com.example.vadapp.vad.eval.SelectParams
import java.util.Locale

fun fmtPct(v: Double) = if (v.isNaN()) "–" else String.format(Locale.US, "%.1f%%", v * 100)
fun fmt3(v: Double) = if (v.isNaN()) "–" else String.format(Locale.US, "%.3f", v)
fun fmtThr(k: Int) = if (k < 0) "–" else String.format(Locale.US, "%.2f", k / 100.0)

/** Simple horizontally scrollable table; the first column is wider and left-aligned. */
@Composable
fun DataTable(
    header: List<String>,
    rows: List<List<String>>,
    modifier: Modifier = Modifier,
    firstColWidth: Dp = 150.dp,
    colWidth: Dp = 76.dp,
    highlightRow: Int = -1,
) {
    val colors = MaterialTheme.colorScheme
    Column(modifier.horizontalScroll(rememberScrollState())) {
        Row(Modifier.background(colors.secondaryContainer)) {
            header.forEachIndexed { i, h ->
                Cell(h, if (i == 0) firstColWidth else colWidth, bold = true, start = i == 0)
            }
        }
        rows.forEachIndexed { r, row ->
            val bg = when {
                r == highlightRow -> colors.primaryContainer
                r % 2 == 1 -> colors.surfaceContainerHigh
                else -> Color.Transparent
            }
            Row(Modifier.background(bg)) {
                row.forEachIndexed { i, v -> Cell(v, if (i == 0) firstColWidth else colWidth, start = i == 0) }
            }
        }
    }
}

@Composable
private fun Cell(text: String, width: Dp, bold: Boolean = false, start: Boolean = false) {
    Text(
        text,
        modifier = Modifier.width(width).padding(horizontal = 4.dp, vertical = 3.dp),
        style = MaterialTheme.typography.labelSmall,
        fontWeight = if (bold) FontWeight.SemiBold else FontWeight.Normal,
        fontFamily = if (start || bold) FontFamily.Default else FontFamily.Monospace,
        textAlign = if (start) TextAlign.Start else TextAlign.End,
        maxLines = if (bold) 3 else 1,
        overflow = TextOverflow.Ellipsis,
    )
}

/** Tables A (threshold choice on dev) and B (test, three modes) of one model. */
@Composable
fun ModelSummaryTables(ms: ModelSummary, params: SelectParams, testOnly: Boolean = false) {
    Column {
        if (testOnly) {
            Text("A. ROC/AUC trên test (quét ngưỡng với merge/drop/pad cố định)", style = MaterialTheme.typography.labelLarge)
            DataTable(
                header = listOf(
                    "Cấu hình", "AUC strict", "AUC collar", "AUC chỉ biên", "AUC chính xác (strict)",
                    "TPR@FPR ${fmtPct(params.fpr1)}", "TPR@FPR ${fmtPct(params.fpr2)}",
                ),
                rows = ms.rows.map { r ->
                    listOf(
                        "${r.config.code} · ${ms.eval.describe(r.config)}", fmt3(r.aucTest[0]), fmt3(r.aucTest[1]),
                        fmt3(r.aucTest[2]), if (r.config == PostConfig.A0) fmt3(r.exactAucTest[0]) else "",
                        fmtPct(r.tprFpr1Test), fmtPct(r.tprFpr2Test),
                    )
                },
                firstColWidth = 190.dp,
                highlightRow = params.config.ordinal,
            )
        } else {
            Text("A. Chọn ngưỡng trên dev (strict)", style = MaterialTheme.typography.labelLarge)
            DataTable(
                header = listOf(
                    "Cấu hình", "Ngưỡng DCF-min", "DCF min", "Ngưỡng F1-max", "F1 max", "Ngưỡng Youden",
                    "Ngưỡng EER", "Ngưỡng MR≤x", "AUC dev", "AUC test", "AUC chính xác dev/test",
                    "TPR@FPR ${fmtPct(params.fpr1)}", "TPR@FPR ${fmtPct(params.fpr2)}", "Ngưỡng dùng",
                ),
                rows = ms.rows.map { r ->
                    listOf(
                        "${r.config.code} · ${ms.eval.describe(r.config)}", fmtThr(r.thrDcf), fmt3(r.dcfMin), fmtThr(r.thrF1),
                        fmt3(r.f1Max), fmtThr(r.thrYouden), fmtThr(r.thrEer), fmtThr(r.thrMr), fmt3(r.aucDev[0]),
                        fmt3(r.aucTest[0]),
                        if (r.config == PostConfig.A0) "${fmt3(r.exactAucDev[0])} / ${fmt3(r.exactAucTest[0])}" else "",
                        fmtPct(r.tprFpr1), fmtPct(r.tprFpr2), fmtThr(r.used),
                    )
                },
                firstColWidth = 190.dp,
                highlightRow = params.config.ordinal,
            )
        }
        Text(
            "B. Test · cấu hình ${ms.config.code} · ngưỡng ${fmtThr(ms.usedK)}",
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.padding(top = 8.dp),
        )
        DataTable(
            header = listOf(
                "Chế độ", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1", "Specificity", "DCF",
                "CI95 F1", "CI95 DCF", "CI95 MR", "CI95 FAR",
            ),
            rows = EvalMode.entries.map { m ->
                val c = ms.test[m.ordinal]
                val ci = ms.ci[m.ordinal]
                fun r3(k: String) = ci[k]?.let { "${fmt3(it.first)}–${fmt3(it.second)}" } ?: "–"
                fun rp(k: String) = ci[k]?.let { "${fmtPct(it.first)}–${fmtPct(it.second)}" } ?: "–"
                listOf(
                    m.label, c.tp.toString(), c.fn.toString(), c.fp.toString(), c.tn.toString(), fmtPct(c.mr),
                    fmtPct(c.far), fmtPct(c.precision), fmtPct(c.recall), fmt3(c.f1), fmtPct(c.specificity),
                    fmt3(c.dcf(params.missWeight)), r3("F1"), r3("DCF"), rp("MR"), rp("FAR"),
                )
            },
            firstColWidth = 170.dp,
        )
        if (ms.groups.isNotEmpty()) {
            Text("Theo nhóm (test, strict)", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 8.dp))
            DataTable(
                header = listOf("Nhóm", "File", "TP", "FN", "FP", "TN", "MR", "FAR", "F1", "DCF"),
                rows = ms.groups.map { g ->
                    val c = g.conf
                    listOf(
                        "${g.kind} ${g.name}", g.files.toString(), c.tp.toString(), c.fn.toString(), c.fp.toString(),
                        c.tn.toString(), fmtPct(c.mr), fmtPct(c.far), fmt3(c.f1), fmt3(c.dcf(params.missWeight)),
                    )
                },
            )
        }
    }
}

/** Table C: one row per model at its chosen threshold (test, strict). */
@Composable
fun ModelComparisonTable(summaries: List<ModelSummary>, params: SelectParams) {
    val c = params.config.ordinal
    DataTable(
        header = listOf("Model", "Cấu hình", "Ngưỡng", "AUC dev", "AUC test", "F1", "MR", "FAR", "DCF", "Precision", "Recall", "Specificity"),
        rows = summaries.map { ms ->
            val t = ms.test[EvalMode.STRICT.ordinal]
            listOf(
                "${ms.model.label} (${ms.eval.source.format.label})", ms.config.code, fmtThr(ms.usedK),
                fmt3(ms.rows[c].aucDev[0]), fmt3(ms.rows[c].aucTest[0]), fmt3(t.f1), fmtPct(t.mr), fmtPct(t.far),
                fmt3(t.dcf(params.missWeight)), fmtPct(t.precision), fmtPct(t.recall), fmtPct(t.specificity),
            )
        },
    )
}
