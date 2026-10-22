package com.example.vadapp.vad.eval

import kotlin.math.roundToInt

/** Ground truth bins are always this long. */
const val BIN_SEC = 0.5f

/**
 * Per-bin labels of one file: 1 = speech, 0 = non-speech, -1 = not annotated.
 * Parsed from lines `start end label`, e.g. `0 0.5 speech` / `0.5 1 no-speech`.
 */
class GroundTruth(val labels: ByteArray) {
    val bins: Int get() = labels.size
    val speechBins: Int get() = labels.count { it.toInt() == 1 }
    val labelledBins: Int get() = labels.count { it >= 0 }
    val durationSec: Float get() = labels.size * BIN_SEC

    companion object {
        fun parse(text: String): GroundTruth {
            val entries = ArrayList<Pair<Int, Byte>>()
            for (raw in text.lineSequence()) {
                val parts = raw.trim().split(Regex("[\\s,;]+"))
                if (parts.size < 3) continue
                val start = parts[0].toFloatOrNull() ?: continue
                val end = parts[1].toFloatOrNull() ?: continue
                val label = parseLabel(parts.drop(2).joinToString(" ")) ?: continue
                // Normally one line per bin, but tolerate lines spanning several bins.
                val first = (start / BIN_SEC).roundToInt()
                val last = ((end / BIN_SEC).roundToInt() - 1).coerceAtLeast(first)
                for (b in first..last) entries += b to label
            }
            val n = (entries.maxOfOrNull { it.first } ?: -1) + 1
            val labels = ByteArray(n) { -1 }
            for ((b, l) in entries) if (b >= 0) labels[b] = l
            return GroundTruth(labels)
        }

        private fun parseLabel(s: String): Byte? {
            val t = s.lowercase().replace("_", "-").replace(" ", "-")
            return when (t) {
                "speech", "1", "s", "sp" -> 1
                "no-speech", "nospeech", "non-speech", "nonspeech", "silence", "0", "ns" -> 0
                else -> if (t.startsWith("no") || t.startsWith("non")) 0 else null
            }
        }
    }
}
