package com.example.vadapp.vad.eval

import com.example.vadapp.vad.AnalysisResult
import com.example.vadapp.vad.ModelSource
import com.example.vadapp.vad.VadModelType
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.security.MessageDigest

/**
 * Model outputs of the dev runs, one small file per (video, model file), so a dataset can be
 * re-evaluated (other ratio / settings / split) or a cancelled run resumed without decoding again.
 */
class ProbCache(root: File) {
    private val dir = File(root, "devcache").apply { mkdirs() }

    private fun file(video: TreeFile, model: VadModelType, source: ModelSource): File {
        val modelStamp = if (source.custom) File(source.path).let { "${it.length()}:${it.lastModified()}" } else ""
        val key = "${video.uriString}|${video.size}|${video.lastModified}|${model.name}|${source.path}|$modelStamp"
        val hash = MessageDigest.getInstance("SHA-1").digest(key.toByteArray())
            .joinToString("") { "%02x".format(it) }
        return File(dir, "$hash.bin")
    }

    fun has(video: TreeFile, model: VadModelType, source: ModelSource) = file(video, model, source).exists()

    fun get(video: TreeFile, model: VadModelType, source: ModelSource): AnalysisResult? {
        val f = file(video, model, source)
        if (!f.exists()) return null
        return runCatching {
            DataInputStream(f.inputStream().buffered()).use { s ->
                check(s.readInt() == VERSION)
                val hop = s.readFloat()
                val offset = s.readFloat()
                val duration = s.readFloat()
                val format = s.readUTF()
                val n = s.readInt()
                val probs = FloatArray(n) { s.readFloat() }
                AnalysisResult(probs, hop, offset, duration, 0, 0, "", format, 0, FloatArray(0), 1f)
            }
        }.getOrNull()
    }

    fun put(video: TreeFile, model: VadModelType, source: ModelSource, r: AnalysisResult) {
        val f = file(video, model, source)
        val tmp = File(f.path + ".tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { s ->
            s.writeInt(VERSION)
            s.writeFloat(r.hopSec)
            s.writeFloat(r.offsetSec)
            s.writeFloat(r.durationSec)
            s.writeUTF(r.format)
            s.writeInt(r.probs.size)
            for (p in r.probs) s.writeFloat(p)
        }
        tmp.renameTo(f)
    }

    fun clear() {
        dir.listFiles()?.forEach { it.delete() }
    }

    private companion object {
        const val VERSION = 1
    }
}
