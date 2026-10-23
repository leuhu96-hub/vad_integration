package com.example.vadapp.vad.eval

import com.example.vadapp.vad.eval.XlsxWriter.Style
import java.io.OutputStream
import java.util.Locale

/**
 * Builds the summary workbook. Black = formula, blue = value computed in the app (bootstrap
 * CI, AUC, per-group counts at the app's threshold), yellow background = editable parameter.
 *
 * Sheet "Quét ngưỡng" holds the raw TP/FN/FP/TN of every (model, set, mode, config, threshold);
 * all rates and the threshold choices are formulas on top of it, so editing a parameter
 * recomputes every table. Block b = ((model·2 + set)·3 + mode)·5 + config occupies rows
 * 2 + 101·b … 2 + 101·b + 100, one row per threshold k/100.
 */
class XlsxReport(
    private val items: List<EvalItem>,
    private val split: Split,
    private val summaries: List<ModelSummary>,
    private val params: SelectParams,
    private val binRatio: Float,
    private val videoRoot: String,
    private val gtRoot: String,
    private val date: String,
    /** Test-only mode: thresholds come from the user (yellow cells), no selection on dev. */
    private val testOnly: Boolean = false,
) {
    private val sum = "'Tổng hợp'!"
    private val sw = "'Quét ngưỡng'!"
    private val configs = PostConfig.entries
    private val cfgList = configs.joinToString(",") { "\"${it.code}\"" }
    private val critList = Criterion.entries.joinToString(",") { "\"${it.excel}\"" }

    // Parameter cells on "Tổng hợp".
    private val pWeight = "\$B\$6"
    private val pConfig = "\$B\$7"
    private val pCrit = "\$B\$8"
    private val pFpr1 = "\$B\$9"
    private val pFpr2 = "\$B\$10"
    private val pMrMax = "\$B\$11"
    private val pAppCfg = "\$B\$13"
    private val pAppCrit = "\$B\$14"
    private val pAppWeight = "\$B\$15"
    private val pCfgIdx = "\$B\$16"

    /** Last data row of "Quét ngưỡng". */
    private val sweepLast get() = blockRow(summaries.size, 0, 0, 0) - 1

    private fun blockRow(m: Int, set: Int, mode: Int, c: Int) = 2 + ((((m * 2 + set) * 3 + mode) * configs.size) + c) * NUM_THR

    fun write(out: OutputStream) {
        val x = XlsxWriter()
        val summary = x.sheet("Tổng hợp")
        val sweep = x.sheet("Quét ngưỡng")
        val splitSheet = x.sheet("Split")
        val groups = x.sheet("Theo nhóm")
        writeSummary(summary)
        writeSweep(sweep)
        writeSplit(splitSheet)
        writeGroups(groups)
        if (summaries.any { it.eval.search != null }) writeSearch(x.sheet("Tìm tham số"))
        x.write(out)
    }

    // ------------------------------------------------------------------ Tổng hợp

    private fun writeSummary(s: XlsxWriter.Sheet) {
        s.width(0, 34.0); s.width(1, 30.0)
        for (c in 2..17) s.width(c, 13.0)
        val nDev = split.isDev.count { it }
        s.str(1, 0, (if (testOnly) "Kết quả test cấu hình user" else "Kết quả đánh giá VAD trên bộ dữ liệu có ground truth") +
            " – chỉ số tính bằng công thức", Style.TITLE)
        s.str(2, 0, "Nguồn: video \"$videoRoot\", GT \"$gtRoot\", chạy $date trên thiết bị. ${items.size} file " +
            "(dev $nDev / test ${items.size - nDev}, tỷ lệ dev ${pct(split.devRatio.toDouble())}), bin GT ${BIN_SEC}s, " +
            "bin là speech khi ≥ ${pct(binRatio.toDouble())} thời lượng có speech.", Style.NOTE)
        s.str(3, 0, "Nền vàng = tham số bạn chỉnh (B6:B11) · chữ đen = công thức · chữ xanh = số nhập từ app " +
            "(bootstrap CI, AUC, đếm theo nhóm; không tính lại trong Excel). Đổi tham số thì mọi bảng tự tính lại; " +
            "hình ROC (PNG) thì không.", Style.NOTE)

        s.str(5, 0, "Tham số", Style.SECTION)
        s.str(6, 0, "Trọng số miss trong DCF (FA = 1 − ô này)"); s.num(6, 1, params.missWeight, Style.PARAM_DEC2)
        s.str(7, 0, "Cấu hình hậu xử lý cho tầng B"); s.str(7, 1, params.config.code, Style.PARAM)
        s.listValidation("B7", configs.map { it.code })
        s.str(8, 0, if (testOnly) "Tiêu chí chọn ngưỡng (không dùng: ngưỡng do user chọn)" else "Tiêu chí chọn ngưỡng trên dev")
        s.str(8, 1, params.criterion.excel, Style.PARAM)
        s.listValidation("B8", Criterion.entries.map { it.excel })
        s.str(9, 0, "Mốc FPR 1 (TPR@FPR)"); s.num(9, 1, params.fpr1, Style.PARAM_PCT)
        s.str(10, 0, "Mốc FPR 2 (TPR@FPR)"); s.num(10, 1, params.fpr2, Style.PARAM_PCT)
        s.str(11, 0, "MR tối đa (cho tiêu chí MR<=x)"); s.num(11, 1, params.mrMax, Style.PARAM_PCT)
        s.str(12, 0, "Tỷ lệ speech tối thiểu trong bin (app)"); s.num(12, 1, binRatio.toDouble(), Style.BLUE_PCT)
        s.str(13, 0, "Cấu hình app đã dùng"); s.str(13, 1, params.config.code, Style.BLUE)
        s.str(14, 0, "Tiêu chí app đã dùng"); s.str(14, 1, params.criterion.excel, Style.BLUE)
        s.str(15, 0, "Trọng số DCF app đã dùng"); s.num(15, 1, params.missWeight, Style.BLUE_DEC2)
        s.str(16, 0, "Chỉ số cấu hình B7 (phụ)", Style.NOTE)
        s.formula(16, 1, "MATCH($pConfig,{$cfgList},0)-1", Style.INT)

        var r = 18
        val strictRowOf = HashMap<Int, Int>()
        val usedCellOf = HashMap<Int, String>()
        val tableARowOf = HashMap<Int, Int>()
        summaries.forEachIndexed { m, ms ->
            val e = ms.eval
            val st = e.settings
            s.str(r, 0, "Model: ${e.model.label} — ${e.source.fileName} (${e.source.format.label}); " +
                "merge ${st.minSilenceMs} ms, drop ${st.minSpeechMs} ms, pad ${st.prePadMs}/${st.postPadMs} ms" +
                (if (e.failed > 0) "; ${e.failed} file lỗi bị bỏ qua" else ""), Style.SECTION)
            val usedRow = r + 1
            val aFirst = r + 7
            if (testOnly) {
                s.str(usedRow, 0, "Ngưỡng dùng (cấu hình user, sửa được)")
                s.num(usedRow, 1, ms.thresholdExact, Style.PARAM_DEC2)
            } else {
                s.str(usedRow, 0, "Ngưỡng dùng (tự chọn trên dev theo B7, B8)")
                s.formula(usedRow, 1, "IFERROR(INDEX(\$Q\$$aFirst:\$Q\$${aFirst + configs.size - 1},$pCfgIdx+1),\"\")", Style.DEC2)
            }
            s.str(r + 2, 0, "Ngưỡng app đã dùng cho CI, theo nhóm, hình")
            s.num(r + 2, 1, ms.thresholdExact, Style.BLUE_DEC2)
            s.str(r + 3, 0, "Kiểm tra")
            val critCheck = if (testOnly) "" else "$pCrit=$pAppCrit,"
            s.formula(r + 3, 1, "IF(AND($pConfig=$pAppCfg,$critCheck$pWeight=$pAppWeight,B$usedRow<>\"\"," +
                "ROUND(N(B$usedRow),2)=ROUND(N(B${r + 2}),2)),\"Khớp\",\"Lệch – CI/nhóm/hình theo app chưa cập nhật\")", Style.BOLD)
            usedCellOf[m] = "B$usedRow"

            // ---- A
            tableARowOf[m] = aFirst
            if (testOnly) {
                writeTestOnlyTableA(s, m, ms, r + 5, aFirst)
            } else {
                s.str(r + 5, 0, "A. Chọn ngưỡng trên dev (strict, qua toàn pipeline; DCF theo trọng số B6)", Style.BOLD)
                val headA = listOf(
                    "Cấu hình", "Mô tả", "Ngưỡng DCF-min (dev)", "DCF min (dev)", "Ngưỡng F1-max (dev)", "F1 max (dev)",
                    "Ngưỡng Youden (dev)", "J max (dev)", "Ngưỡng EER (dev)", "Ngưỡng MR<=x (dev)", "AUC dev (lưới ngưỡng)",
                    "AUC test (lưới ngưỡng)", "AUC chính xác dev", "AUC chính xác test", "TPR@FPR1 (dev)", "TPR@FPR2 (dev)",
                    "Ngưỡng dùng (theo B8)",
                )
                headA.forEachIndexed { c, h -> s.str(r + 6, c, h, Style.HEADER) }
                configs.forEachIndexed { c, cfg ->
                    val row = aFirst + c
                    val r0 = blockRow(m, 0, 0, c)
                    val r1 = r0 + NUM_THR - 1
                    fun rng(col: String) = "$sw\$$col\$$r0:\$$col\$$r1"
                    fun argOpt(fn: String, col: String) =
                        "IF(COUNT(${rng(col)})=0,\"\",INDEX(${rng("E")},MATCH($fn(${rng(col)}),${rng(col)},0)))"
                    val row0 = ms.rows[c]
                    s.str(row, 0, cfg.code)
                    s.str(row, 1, e.describe(cfg))
                    s.formula(row, 2, argOpt("MIN", "P"), Style.DEC2)
                    s.formula(row, 3, "IF(COUNT(${rng("P")})=0,\"\",MIN(${rng("P")}))", Style.DEC3)
                    s.formula(row, 4, argOpt("MAX", "N"), Style.DEC2)
                    s.formula(row, 5, "IF(COUNT(${rng("N")})=0,\"\",MAX(${rng("N")}))", Style.DEC3)
                    s.formula(row, 6, argOpt("MAX", "Q"), Style.DEC2)
                    s.formula(row, 7, "IF(COUNT(${rng("Q")})=0,\"\",MAX(${rng("Q")}))", Style.DEC3)
                    s.formula(row, 8, argOpt("MIN", "R"), Style.DEC2)
                    s.formula(row, 9, argOpt("MIN", "S"), Style.DEC2)
                    s.num(row, 10, row0.aucDev[0], Style.BLUE_DEC3)
                    s.num(row, 11, row0.aucTest[0], Style.BLUE_DEC3)
                    s.num(row, 12, row0.exactAucDev[0], Style.BLUE_DEC3)
                    s.num(row, 13, row0.exactAucTest[0], Style.BLUE_DEC3)
                    s.formula(row, 14, "IFERROR(SUMPRODUCT(MAX((${rng("K")}<=$pFpr1)*${rng("M")})),\"\")", Style.PCT)
                    s.formula(row, 15, "IFERROR(SUMPRODUCT(MAX((${rng("K")}<=$pFpr2)*${rng("M")})),\"\")", Style.PCT)
                    s.formula(row, 16, "IFERROR(CHOOSE(MATCH($pCrit,{$critList},0),C$row,E$row,G$row,I$row,J$row),\"\")", Style.DEC2)
                }
            }

            // ---- B
            val bTitle = aFirst + configs.size + 1
            s.str(bTitle, 0, "B. Tầng B trên test (cấu hình B7, ngưỡng B$usedRow)", Style.BOLD)
            val headB = listOf(
                "Chế độ", "Vùng bin", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1",
                "Specificity", "DCF", "CI95 F1 (app)", "CI95 DCF (app)", "CI95 MR (app)", "CI95 FAR (app)", "Dòng tra (Quét ngưỡng)",
            )
            headB.forEachIndexed { c, h -> s.str(bTitle + 1, c, h, Style.HEADER) }
            EvalMode.entries.forEachIndexed { md, mode ->
                val row = bTitle + 2 + md
                if (md == 0) strictRowOf[m] = row
                val base = blockRow(m, 1, md, 0)
                s.str(row, 0, mode.label)
                s.str(row, 1, mode.short)
                s.formula(row, 17, "IFERROR($base+$pCfgIdx*$NUM_THR+ROUND(\$B\$$usedRow*100,0),\"\")", Style.INT)
                listOf("F", "G", "H", "I").forEachIndexed { i, col ->
                    s.formula(row, 2 + i, "IFERROR(INDEX($sw\$$col\$1:\$$col\$$sweepLast,\$R$row),\"\")", Style.INT)
                }
                s.formula(row, 6, "IFERROR(D$row/(C$row+D$row),\"\")", Style.PCT)
                s.formula(row, 7, "IFERROR(E$row/(E$row+F$row),\"\")", Style.PCT)
                s.formula(row, 8, "IFERROR(C$row/(C$row+E$row),\"\")", Style.PCT)
                s.formula(row, 9, "IFERROR(C$row/(C$row+D$row),\"\")", Style.PCT)
                s.formula(row, 10, "IFERROR(2*C$row/(2*C$row+D$row+E$row),\"\")", Style.DEC3)
                s.formula(row, 11, "IFERROR(F$row/(F$row+E$row),\"\")", Style.PCT)
                s.formula(row, 12, "IFERROR($pWeight*G$row+(1-$pWeight)*H$row,\"\")", Style.DEC3)
                val ci = ms.ci[md]
                s.str(row, 13, ci["F1"]?.let { "${d3(it.first)}–${d3(it.second)}" }, Style.BLUE)
                s.str(row, 14, ci["DCF"]?.let { "${d3(it.first)}–${d3(it.second)}" }, Style.BLUE)
                s.str(row, 15, ci["MR"]?.let { "${pct(it.first)}–${pct(it.second)}" }, Style.BLUE)
                s.str(row, 16, ci["FAR"]?.let { "${pct(it.first)}–${pct(it.second)}" }, Style.BLUE)
            }
            r = bTitle + 2 + EvalMode.entries.size + 2
        }

        // ---- C
        s.str(r, 0, "C. So sánh model (test, strict, cấu hình B7)", Style.BOLD)
        val headC = listOf(
            "Model", "File model", "Cấu hình", "Ngưỡng dùng", "AUC dev (lưới)", "AUC test (lưới)", "AUC chính xác test (A0)",
            "F1", "MR", "FAR", "DCF", "Precision", "Recall", "Specificity",
        )
        headC.forEachIndexed { c, h -> s.str(r + 1, c, h, Style.HEADER) }
        summaries.forEachIndexed { m, ms ->
            val row = r + 2 + m
            val sr = strictRowOf.getValue(m)
            val a0 = tableARowOf.getValue(m)
            s.str(row, 0, ms.model.label)
            s.str(row, 1, "${ms.eval.source.fileName} (${ms.eval.source.format.label})")
            s.formula(row, 2, pConfig)
            s.formula(row, 3, usedCellOf.getValue(m), Style.DEC2)
            // Test-only table A has AUC test (strict) in column C and no dev.
            if (!testOnly) s.formula(row, 4, "IFERROR(INDEX(\$K\$$a0:\$K\$${a0 + configs.size - 1},$pCfgIdx+1),\"\")", Style.DEC3)
            val aucCol = if (testOnly) "C" else "L"
            s.formula(row, 5, "IFERROR(INDEX(\$$aucCol\$$a0:\$$aucCol\$${a0 + configs.size - 1},$pCfgIdx+1),\"\")", Style.DEC3)
            s.num(row, 6, ms.rows[0].exactAucTest[0], Style.BLUE_DEC3)
            s.formula(row, 7, "K$sr", Style.DEC3)
            s.formula(row, 8, "G$sr", Style.PCT)
            s.formula(row, 9, "H$sr", Style.PCT)
            s.formula(row, 10, "M$sr", Style.DEC3)
            s.formula(row, 11, "I$sr", Style.PCT)
            s.formula(row, 12, "J$sr", Style.PCT)
            s.formula(row, 13, "L$sr", Style.PCT)
        }
    }

    /** Test-only table A: ROC/AUC on test per configuration (merge/drop/pad fixed, threshold swept). */
    private fun writeTestOnlyTableA(s: XlsxWriter.Sheet, m: Int, ms: ModelSummary, title: Int, aFirst: Int) {
        s.str(title, 0, "A. ROC/AUC trên test (quét ngưỡng với merge/drop/pad của từng cấu hình)", Style.BOLD)
        val head = listOf(
            "Cấu hình", "Mô tả", "AUC test strict", "AUC test collar", "AUC test chỉ biên", "AUC chính xác test (A0)",
            "TPR@FPR1 (test)", "TPR@FPR2 (test)",
        )
        head.forEachIndexed { c, h -> s.str(title + 1, c, h, Style.HEADER) }
        configs.forEachIndexed { c, cfg ->
            val row = aFirst + c
            val r0 = blockRow(m, 1, 0, c)
            val r1 = r0 + NUM_THR - 1
            fun rng(col: String) = "$sw\$$col\$$r0:\$$col\$$r1"
            val rr = ms.rows[c]
            s.str(row, 0, cfg.code)
            s.str(row, 1, ms.eval.describe(cfg))
            s.num(row, 2, rr.aucTest[0], Style.BLUE_DEC3)
            s.num(row, 3, rr.aucTest[1], Style.BLUE_DEC3)
            s.num(row, 4, rr.aucTest[2], Style.BLUE_DEC3)
            s.num(row, 5, rr.exactAucTest[0], Style.BLUE_DEC3)
            s.formula(row, 6, "IFERROR(SUMPRODUCT(MAX((${rng("K")}<=$pFpr1)*${rng("M")})),\"\")", Style.PCT)
            s.formula(row, 7, "IFERROR(SUMPRODUCT(MAX((${rng("K")}<=$pFpr2)*${rng("M")})),\"\")", Style.PCT)
        }
    }

    // ------------------------------------------------------------------ Quét ngưỡng

    private fun writeSweep(s: XlsxWriter.Sheet) {
        val head = listOf(
            "Model", "Cấu hình", "Tập", "Chế độ", "Ngưỡng", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision",
            "Recall", "F1", "Specificity", "DCF", "Youden J", "|MR − FAR|", "FAR nếu MR<=x",
        )
        head.forEachIndexed { c, h -> s.str(1, c, h, Style.HEADER) }
        s.freeze(1)
        s.width(0, 14.0); s.width(3, 8.0)
        val w = "$sum$pWeight"
        val mrMax = "$sum$pMrMax"
        summaries.forEachIndexed { m, ms ->
            for (set in 0..1) {
                val sweep = if (set == 0) ms.eval.devSweep else ms.eval.testSweep
                for (md in EvalMode.entries.indices) for (c in configs.indices) {
                    val r0 = blockRow(m, set, md, c)
                    for (k in 0 until NUM_THR) {
                        val r = r0 + k
                        val conf = Metrics.conf(sweep, c, k, md)
                        s.str(r, 0, ms.model.label)
                        s.str(r, 1, configs[c].code)
                        s.str(r, 2, if (set == 0) "dev" else "test")
                        s.str(r, 3, EvalMode.entries[md].short)
                        s.num(r, 4, k / 100.0, Style.DEC2)
                        s.num(r, 5, conf.tp.toDouble())
                        s.num(r, 6, conf.fn.toDouble())
                        s.num(r, 7, conf.fp.toDouble())
                        s.num(r, 8, conf.tn.toDouble())
                        s.formula(r, 9, "IF(F$r+G$r=0,\"\",G$r/(F$r+G$r))", Style.PCT)
                        s.formula(r, 10, "IF(H$r+I$r=0,\"\",H$r/(H$r+I$r))", Style.PCT)
                        s.formula(r, 11, "IF(F$r+H$r=0,\"\",F$r/(F$r+H$r))", Style.PCT)
                        s.formula(r, 12, "IF(F$r+G$r=0,\"\",F$r/(F$r+G$r))", Style.PCT)
                        s.formula(r, 13, "IF(2*F$r+G$r+H$r=0,\"\",2*F$r/(2*F$r+G$r+H$r))", Style.DEC3)
                        s.formula(r, 14, "IF(I$r+H$r=0,\"\",I$r/(I$r+H$r))", Style.PCT)
                        s.formula(r, 15, "IF(OR(J$r=\"\",K$r=\"\"),\"\",$w*J$r+(1-$w)*K$r)", Style.DEC3)
                        s.formula(r, 16, "IF(OR(M$r=\"\",K$r=\"\"),\"\",M$r-K$r)", Style.DEC3)
                        s.formula(r, 17, "IF(OR(J$r=\"\",K$r=\"\"),\"\",ABS(J$r-K$r))", Style.DEC3)
                        s.formula(r, 18, "IF(OR(J$r=\"\",K$r=\"\"),\"\",IF(J$r<=$mrMax,K$r,\"\"))", Style.PCT)
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ Split

    private fun writeSplit(s: XlsxWriter.Sheet) {
        val head = listOf(
            "Video", "Đường dẫn", "Quality", "Category", "File GT", "Tập", "Số bin GT (có nhãn)", "Bin speech",
            "Tỷ lệ speech", "Thời lượng GT (s)",
        )
        head.forEachIndexed { c, h -> s.str(1, c, h, Style.HEADER) }
        s.freeze(1)
        s.width(0, 28.0); s.width(1, 40.0); s.width(4, 30.0)
        items.forEachIndexed { i, it ->
            val r = i + 2
            s.str(r, 0, it.name)
            s.str(r, 1, it.video.relPath)
            s.str(r, 2, it.quality)
            s.str(r, 3, it.category)
            s.str(r, 4, it.gtFile.relPath)
            s.str(r, 5, if (split.isDev[i]) "dev" else "test")
            s.num(r, 6, it.gt.labelledBins.toDouble())
            s.num(r, 7, it.gt.speechBins.toDouble())
            s.formula(r, 8, "IF(G$r=0,\"\",H$r/G$r)", Style.PCT)
            s.formula(r, 9, "G$r*$BIN_SEC", Style.DEC2)
        }
        val last = items.size + 1
        fun col(c: String) = "\$$c\$2:\$$c\$$last"
        val c0 = 11
        val headCell = listOf(
            "Quality", "Category", "Dev (file)", "Test (file)", "Dev (s)", "Test (s)", "Dev / tổng thời lượng",
            "Speech dev", "Speech test",
        )
        headCell.forEachIndexed { c, h -> s.str(1, c0 + c, h, Style.HEADER) }
        for (c in 0 until headCell.size) s.width(c0 + c, 12.0)
        val rows = items.map { it.quality to it.category }.distinct()
            .sortedWith(compareBy({ it.first }, { it.second })) + listOf("*" to "*")
        rows.forEachIndexed { i, (q, cat) ->
            val r = i + 2
            val total = q == "*"
            s.str(r, c0, if (total) "Tổng" else q, if (total) Style.BOLD else Style.DEFAULT)
            s.str(r, c0 + 1, if (total) "" else cat)
            val crit = if (total) "" else ",${col("C")},${XlsxWriter.ref(r, c0)},${col("D")},${XlsxWriter.ref(r, c0 + 1)}"
            s.formula(r, c0 + 2, "COUNTIFS(${col("F")},\"dev\"$crit)", Style.INT)
            s.formula(r, c0 + 3, "COUNTIFS(${col("F")},\"test\"$crit)", Style.INT)
            s.formula(r, c0 + 4, "SUMIFS(${col("J")},${col("F")},\"dev\"$crit)", Style.DEC2)
            s.formula(r, c0 + 5, "SUMIFS(${col("J")},${col("F")},\"test\"$crit)", Style.DEC2)
            val dv = XlsxWriter.ref(r, c0 + 4)
            val tv = XlsxWriter.ref(r, c0 + 5)
            s.formula(r, c0 + 6, "IFERROR($dv/($dv+$tv),\"\")", Style.PCT)
            s.formula(r, c0 + 7, "IFERROR(SUMIFS(${col("H")},${col("F")},\"dev\"$crit)/SUMIFS(${col("G")},${col("F")},\"dev\"$crit),\"\")", Style.PCT)
            s.formula(r, c0 + 8, "IFERROR(SUMIFS(${col("H")},${col("F")},\"test\"$crit)/SUMIFS(${col("G")},${col("F")},\"test\"$crit),\"\")", Style.PCT)
        }
    }

    // ------------------------------------------------------------------ Theo nhóm

    private fun writeGroups(s: XlsxWriter.Sheet) {
        s.str(1, 0, "Theo nhóm – tập test, strict, cấu hình và ngưỡng app đã dùng (đếm là số xanh, không tự tính lại khi đổi tham số)", Style.NOTE)
        val head = listOf(
            "Model", "Loại", "Nhóm", "Số file test", "Cấu hình", "Ngưỡng", "TP", "FN", "FP", "TN", "MR", "FAR",
            "Precision", "Recall", "F1", "Specificity", "DCF",
        )
        head.forEachIndexed { c, h -> s.str(2, c, h, Style.HEADER) }
        s.freeze(2)
        s.width(0, 14.0)
        var r = 3
        for (ms in summaries) for (g in ms.groups) {
            s.str(r, 0, ms.model.label)
            s.str(r, 1, g.kind)
            s.str(r, 2, g.name)
            s.num(r, 3, g.files.toDouble(), Style.BLUE_INT)
            s.str(r, 4, ms.config.code, Style.BLUE)
            s.num(r, 5, ms.thresholdExact, Style.BLUE_DEC2)
            s.num(r, 6, g.conf.tp.toDouble(), Style.BLUE_INT)
            s.num(r, 7, g.conf.fn.toDouble(), Style.BLUE_INT)
            s.num(r, 8, g.conf.fp.toDouble(), Style.BLUE_INT)
            s.num(r, 9, g.conf.tn.toDouble(), Style.BLUE_INT)
            s.formula(r, 10, "IFERROR(H$r/(G$r+H$r),\"\")", Style.PCT)
            s.formula(r, 11, "IFERROR(I$r/(I$r+J$r),\"\")", Style.PCT)
            s.formula(r, 12, "IFERROR(G$r/(G$r+I$r),\"\")", Style.PCT)
            s.formula(r, 13, "IFERROR(G$r/(G$r+H$r),\"\")", Style.PCT)
            s.formula(r, 14, "IFERROR(2*G$r/(2*G$r+H$r+I$r),\"\")", Style.DEC3)
            s.formula(r, 15, "IFERROR(J$r/(J$r+I$r),\"\")", Style.PCT)
            s.formula(r, 16, "IFERROR($sum$pWeight*K$r+(1-$sum$pWeight)*L$r,\"\")", Style.DEC3)
            r++
        }
    }

    // ------------------------------------------------------------------ Tìm tham số

    private fun writeSearch(s: XlsxWriter.Sheet) {
        s.str(1, 0, "Tìm bộ hậu xử lý trên dev (strict): mỗi tổ hợp với ngưỡng tốt nhất của nó, sắp theo mục tiêu. " +
            "Việc chọn tổ hợp dùng tiêu chí/trọng số lúc tìm (cột Tiêu chí) và không tự tính lại khi đổi tham số; " +
            "số đếm là số xanh, MR/FAR/F1/DCF là công thức.", Style.NOTE)
        val head = listOf(
            "Model", "Tiêu chí", "Hạng", "Merge (ms)", "Drop (ms)", "Pre pad (ms)", "Post pad (ms)", "Ngưỡng tốt nhất",
            "Mục tiêu", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1", "DCF",
        )
        head.forEachIndexed { c, h -> s.str(2, c, h, Style.HEADER) }
        s.freeze(2)
        s.width(0, 14.0); s.width(1, 14.0)
        var r = 3
        val w = "$sum$pWeight"
        for (ms in summaries) {
            val res = ms.eval.search ?: continue
            val crit = res.criterion.excel + if (res.criterion == Criterion.DCF_MIN) " w=${res.missWeight}" else ""
            res.ranked().forEachIndexed { rank, e ->
                val st = e.settings
                val c = e.conf[EvalMode.STRICT.ordinal]
                s.str(r, 0, ms.model.label)
                s.str(r, 1, crit)
                s.num(r, 2, rank + 1.0, Style.INT)
                s.num(r, 3, st.minSilenceMs.toDouble(), Style.BLUE_INT)
                s.num(r, 4, st.minSpeechMs.toDouble(), Style.BLUE_INT)
                s.num(r, 5, st.prePadMs.toDouble(), Style.BLUE_INT)
                s.num(r, 6, st.postPadMs.toDouble(), Style.BLUE_INT)
                s.num(r, 7, if (e.bestK >= 0) e.bestK / 100.0 else null, Style.BLUE_DEC2)
                s.num(r, 8, res.value(e), Style.BLUE_DEC3)
                s.num(r, 9, c.tp.toDouble(), Style.BLUE_INT)
                s.num(r, 10, c.fn.toDouble(), Style.BLUE_INT)
                s.num(r, 11, c.fp.toDouble(), Style.BLUE_INT)
                s.num(r, 12, c.tn.toDouble(), Style.BLUE_INT)
                s.formula(r, 13, "IFERROR(K$r/(J$r+K$r),\"\")", Style.PCT)
                s.formula(r, 14, "IFERROR(L$r/(L$r+M$r),\"\")", Style.PCT)
                s.formula(r, 15, "IFERROR(J$r/(J$r+L$r),\"\")", Style.PCT)
                s.formula(r, 16, "IFERROR(J$r/(J$r+K$r),\"\")", Style.PCT)
                s.formula(r, 17, "IFERROR(2*J$r/(2*J$r+K$r+L$r),\"\")", Style.DEC3)
                s.formula(r, 18, "IFERROR($w*N$r+(1-$w)*O$r,\"\")", Style.DEC3)
                r++
            }
        }
    }

    private fun d3(v: Double) = String.format(Locale.US, "%.3f", v)
    private fun pct(v: Double) = String.format(Locale.US, "%.1f%%", v * 100)
}
