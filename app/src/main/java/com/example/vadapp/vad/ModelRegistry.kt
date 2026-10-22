package com.example.vadapp.vad

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.InputStream

enum class ModelFormat(val label: String) {
    ONNX("ONNX"),
    TFLITE("TFLite");

    companion object {
        /** Same sniffing as native DetectFormat: "TFL3" flatbuffer id, or an ONNX protobuf. */
        fun detect(header: ByteArray): ModelFormat? = when {
            header.size >= 8 && String(header, 4, 4, Charsets.US_ASCII) == "TFL3" -> TFLITE
            header.isNotEmpty() && header[0] == 0x08.toByte() -> ONNX
            else -> null
        }
    }
}

/** A concrete model file: bundled in assets ("asset:...") or imported by the user. */
data class ModelSource(
    val path: String,
    val format: ModelFormat,
    val fileName: String,
    val custom: Boolean,
) {
    val label: String get() = if (custom) fileName else format.label
}

/**
 * Finds the model files available for each model type. Whatever format a file is in,
 * that is the format used: the bundled list is built from what actually exists in
 * assets/models, and imported files are classified by their content.
 */
class ModelRegistry(private val context: Context) {

    private val customRoot = File(context.filesDir, "models")

    fun sources(type: VadModelType): List<ModelSource> = bundled(type) + imported(type)

    private fun bundled(type: VadModelType): List<ModelSource> {
        val names = context.assets.list("models").orEmpty().toSet()
        return listOf("onnx", "tflite")
            .map { "${type.assetBase}.$it" }
            .filter { it in names }
            .mapNotNull { name ->
                val format = context.assets.open("models/$name").use { detect(it) } ?: return@mapNotNull null
                ModelSource("asset:models/$name", format, name, custom = false)
            }
    }

    private fun imported(type: VadModelType): List<ModelSource> =
        File(customRoot, type.name.lowercase()).listFiles().orEmpty()
            .sortedBy { it.name }
            .mapNotNull { f ->
                val format = f.inputStream().use { detect(it) } ?: return@mapNotNull null
                ModelSource(f.absolutePath, format, f.name, custom = true)
            }

    /** Copies a user-picked model into app storage. Throws if it is neither ONNX nor TFLite. */
    fun import(type: VadModelType, uri: Uri): ModelSource {
        val resolver = context.contentResolver
        val format = resolver.openInputStream(uri)?.use { detect(it) }
            ?: throw IllegalArgumentException("File không phải model ONNX hoặc TFLite")
        val rawName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
            ?: "model"
        // Keep the user's name but make the extension match the detected format.
        val base = rawName.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9._-]"), "_")
        val name = "$base.${if (format == ModelFormat.TFLITE) "tflite" else "onnx"}"
        val dir = File(customRoot, type.name.lowercase()).apply { mkdirs() }
        val dest = File(dir, name)
        resolver.openInputStream(uri)!!.use { input -> dest.outputStream().use { input.copyTo(it) } }
        return ModelSource(dest.absolutePath, format, name, custom = true)
    }

    fun delete(source: ModelSource) {
        if (source.custom) File(source.path).delete()
    }

    private fun detect(input: InputStream): ModelFormat? {
        val header = ByteArray(8)
        var n = 0
        while (n < header.size) {
            val r = input.read(header, n, header.size - n)
            if (r < 0) break
            n += r
        }
        return ModelFormat.detect(header.copyOf(n))
    }
}
