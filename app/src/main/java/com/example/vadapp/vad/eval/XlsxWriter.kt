package com.example.vadapp.vad.eval

import java.io.OutputStream
import java.util.TreeMap
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Minimal .xlsx (SpreadsheetML) writer: numbers, inline strings, formulas, a fixed style
 * table, column widths, frozen header rows and list validations. Formulas are written
 * without cached values and the workbook asks for a full recalculation on load.
 */
class XlsxWriter {

    /** Indexes into the cellXfs table in [STYLES]. */
    object Style {
        const val DEFAULT = 0
        const val BOLD = 1
        const val HEADER = 2
        const val PARAM = 3
        const val PARAM_PCT = 4
        const val PCT = 5
        const val DEC3 = 6
        const val DEC2 = 7
        const val INT = 8
        const val BLUE = 9
        const val BLUE_DEC3 = 10
        const val BLUE_PCT = 11
        const val TITLE = 12
        const val NOTE = 13
        const val BLUE_DEC2 = 14
        const val PARAM_DEC2 = 15
        const val BLUE_INT = 16
        const val SECTION = 17
    }

    sealed class Cell(val style: Int) {
        class Num(val v: Double, style: Int) : Cell(style)
        class Str(val v: String, style: Int) : Cell(style)
        class Formula(val f: String, style: Int) : Cell(style)
    }

    class Sheet(val name: String) {
        internal val rows = TreeMap<Int, TreeMap<Int, Cell>>()
        internal val widths = TreeMap<Int, Double>()
        internal val validations = ArrayList<Pair<String, List<String>>>()
        internal var frozenRows = 0

        /** [row] is 1-based, [col] 0-based (A = 0). */
        fun num(row: Int, col: Int, v: Double?, style: Int = Style.DEFAULT) {
            if (v == null || v.isNaN() || v.isInfinite()) return
            put(row, col, Cell.Num(v, style))
        }

        fun str(row: Int, col: Int, v: String?, style: Int = Style.DEFAULT) {
            if (v.isNullOrEmpty()) return
            put(row, col, Cell.Str(v, style))
        }

        fun formula(row: Int, col: Int, f: String, style: Int = Style.DEFAULT) = put(row, col, Cell.Formula(f, style))

        fun width(col: Int, chars: Double) { widths[col] = chars }
        fun freeze(rows: Int) { frozenRows = rows }
        fun listValidation(ref: String, values: List<String>) { validations += ref to values }

        private fun put(row: Int, col: Int, c: Cell) {
            rows.getOrPut(row) { TreeMap() }[col] = c
        }
    }

    private val sheets = ArrayList<Sheet>()

    fun sheet(name: String) = Sheet(name).also { sheets += it }

    fun write(out: OutputStream) {
        ZipOutputStream(out).use { zip ->
            fun entry(name: String, body: String) {
                zip.putNextEntry(ZipEntry(name))
                zip.write(body.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
            entry("[Content_Types].xml", buildString {
                append(XML_HEAD)
                append("<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">")
                append("<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>")
                append("<Default Extension=\"xml\" ContentType=\"application/xml\"/>")
                append("<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>")
                append("<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>")
                sheets.indices.forEach {
                    append("<Override PartName=\"/xl/worksheets/sheet${it + 1}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>")
                }
                append("</Types>")
            })
            entry("_rels/.rels", XML_HEAD +
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" +
                "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>" +
                "</Relationships>")
            entry("xl/workbook.xml", buildString {
                append(XML_HEAD)
                append("<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>")
                sheets.forEachIndexed { i, s ->
                    append("<sheet name=\"${esc(s.name)}\" sheetId=\"${i + 1}\" r:id=\"rId${i + 1}\"/>")
                }
                append("</sheets><calcPr calcId=\"191029\" fullCalcOnLoad=\"1\"/></workbook>")
            })
            entry("xl/_rels/workbook.xml.rels", buildString {
                append(XML_HEAD)
                append("<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">")
                sheets.indices.forEach {
                    append("<Relationship Id=\"rId${it + 1}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet${it + 1}.xml\"/>")
                }
                append("<Relationship Id=\"rId${sheets.size + 1}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>")
                append("</Relationships>")
            })
            entry("xl/styles.xml", STYLES)
            sheets.forEachIndexed { i, s ->
                zip.putNextEntry(ZipEntry("xl/worksheets/sheet${i + 1}.xml"))
                val w = zip.bufferedWriter(Charsets.UTF_8)
                writeSheet(s, w)
                w.flush()
                zip.closeEntry()
            }
        }
    }

    private fun writeSheet(s: Sheet, w: java.io.Writer) {
        w.write(XML_HEAD)
        w.write("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">")
        if (s.frozenRows > 0) {
            w.write("<sheetViews><sheetView workbookViewId=\"0\"><pane ySplit=\"${s.frozenRows}\" " +
                "topLeftCell=\"A${s.frozenRows + 1}\" activePane=\"bottomLeft\" state=\"frozen\"/></sheetView></sheetViews>")
        }
        if (s.widths.isNotEmpty()) {
            w.write("<cols>")
            for ((c, width) in s.widths) w.write("<col min=\"${c + 1}\" max=\"${c + 1}\" width=\"$width\" customWidth=\"1\"/>")
            w.write("</cols>")
        }
        w.write("<sheetData>")
        for ((r, cells) in s.rows) {
            w.write("<row r=\"$r\">")
            for ((c, cell) in cells) {
                val ref = "${col(c)}$r"
                val st = if (cell.style != 0) " s=\"${cell.style}\"" else ""
                when (cell) {
                    is Cell.Num -> w.write("<c r=\"$ref\"$st><v>${fmt(cell.v)}</v></c>")
                    is Cell.Str -> w.write("<c r=\"$ref\"$st t=\"inlineStr\"><is><t xml:space=\"preserve\">${esc(cell.v)}</t></is></c>")
                    is Cell.Formula -> w.write("<c r=\"$ref\"$st><f>${esc(cell.f)}</f></c>")
                }
            }
            w.write("</row>")
        }
        w.write("</sheetData>")
        if (s.validations.isNotEmpty()) {
            w.write("<dataValidations count=\"${s.validations.size}\">")
            for ((ref, values) in s.validations) {
                w.write("<dataValidation type=\"list\" allowBlank=\"1\" showErrorMessage=\"1\" sqref=\"$ref\">" +
                    "<formula1>${esc("\"" + values.joinToString(",") + "\"")}</formula1></dataValidation>")
            }
            w.write("</dataValidations>")
        }
        w.write("</worksheet>")
    }

    companion object {
        private const val XML_HEAD = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"

        fun col(c: Int): String {
            var n = c + 1
            val sb = StringBuilder()
            while (n > 0) {
                val m = (n - 1) % 26
                sb.insert(0, ('A' + m))
                n = (n - 1) / 26
            }
            return sb.toString()
        }

        fun ref(row: Int, c: Int, absolute: Boolean = false) =
            if (absolute) "\$${col(c)}\$$row" else "${col(c)}$row"

        private fun fmt(v: Double) =
            if (v == Math.rint(v) && kotlin.math.abs(v) < 1e15) v.toLong().toString() else v.toString()

        private fun esc(s: String) = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")

        // fonts: 0 normal, 1 bold, 2 blue, 3 title, 4 note, 5 bold section
        // fills: 0 none, 1 gray125 (required), 2 yellow (parameters), 3 header
        // numFmts: 164 0.0%, 165 0.000, 166 0.00, builtin 1 = 0
        private val STYLES = XML_HEAD +
            "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">" +
            "<numFmts count=\"3\"><numFmt numFmtId=\"164\" formatCode=\"0.0%\"/><numFmt numFmtId=\"165\" formatCode=\"0.000\"/>" +
            "<numFmt numFmtId=\"166\" formatCode=\"0.00\"/></numFmts>" +
            "<fonts count=\"6\">" +
            "<font><sz val=\"11\"/><name val=\"Calibri\"/></font>" +
            "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font>" +
            "<font><sz val=\"11\"/><color rgb=\"FF0000FF\"/><name val=\"Calibri\"/></font>" +
            "<font><b/><sz val=\"14\"/><name val=\"Calibri\"/></font>" +
            "<font><i/><sz val=\"10\"/><color rgb=\"FF595959\"/><name val=\"Calibri\"/></font>" +
            "<font><b/><sz val=\"12\"/><color rgb=\"FF1F3864\"/><name val=\"Calibri\"/></font>" +
            "</fonts>" +
            "<fills count=\"4\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill>" +
            "<fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFFFF2CC\"/><bgColor indexed=\"64\"/></patternFill></fill>" +
            "<fill><patternFill patternType=\"solid\"><fgColor rgb=\"FFD9E1F2\"/><bgColor indexed=\"64\"/></patternFill></fill></fills>" +
            "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>" +
            "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>" +
            "<cellXfs count=\"18\">" +
            xf(0, 0, 0) +                                   // 0 default
            xf(0, 1, 0) +                                   // 1 bold
            xf(0, 1, 3, wrap = true) +                      // 2 header
            xf(0, 0, 2) +                                   // 3 param
            xf(164, 0, 2) +                                 // 4 param %
            xf(164, 0, 0) +                                 // 5 %
            xf(165, 0, 0) +                                 // 6 0.000
            xf(166, 0, 0) +                                 // 7 0.00
            xf(1, 0, 0) +                                   // 8 0
            xf(0, 2, 0) +                                   // 9 blue
            xf(165, 2, 0) +                                 // 10 blue 0.000
            xf(164, 2, 0) +                                 // 11 blue %
            xf(0, 3, 0) +                                   // 12 title
            xf(0, 4, 0) +                                   // 13 note
            xf(166, 2, 0) +                                 // 14 blue 0.00
            xf(166, 0, 2) +                                 // 15 param 0.00
            xf(1, 2, 0) +                                   // 16 blue 0
            xf(0, 5, 0) +                                   // 17 section
            "</cellXfs>" +
            "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>" +
            "</styleSheet>"

        private fun xf(numFmt: Int, font: Int, fill: Int, wrap: Boolean = false) =
            "<xf numFmtId=\"$numFmt\" fontId=\"$font\" fillId=\"$fill\" borderId=\"0\" xfId=\"0\"" +
                (if (numFmt != 0) " applyNumberFormat=\"1\"" else "") +
                (if (font != 0) " applyFont=\"1\"" else "") +
                (if (fill != 0) " applyFill=\"1\"" else "") +
                (if (wrap) " applyAlignment=\"1\"><alignment wrapText=\"1\" vertical=\"center\"/></xf>" else "/>")
    }
}
