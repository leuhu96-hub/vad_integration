package com.example.vadapp

import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.vadapp.vad.ModelRegistry
import com.example.vadapp.vad.NativeVad
import com.example.vadapp.vad.VadModelType
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Runs every available model file (each bundled .onnx/.tflite) on every media file in
 * <filesDir>/vadtest/ and writes <name>.<model>.<format>.json next to it (probabilities + default segments), so results can be
 * compared against a desktop reference:
 *
 *   ./gradlew installDebug installDebugAndroidTest
 *   adb push clip.mp4 /data/local/tmp/
 *   adb shell run-as com.example.vadapp sh -c 'mkdir -p files/vadtest && cp /data/local/tmp/clip.mp4 files/vadtest/'
 *   adb shell am instrument -w com.example.vadapp.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class NativeVadTest {

    @Test
    fun analyzeMediaFiles() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.filesDir, "vadtest")
        val inputs = dir.listFiles { f -> !f.name.endsWith(".json") }.orEmpty()
        assumeTrue("no media in $dir", inputs.isNotEmpty())

        val registry = ModelRegistry(ctx)
        for (file in inputs) {
            for (model in VadModelType.entries) for (source in registry.sources(model)) {
                val result = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
                    NativeVad.analyze(
                        ctx.assets, model.id, source.path, model.extraAsset, pfd.fd, 0, file.length(), 4,
                    ) { true }
                }
                assertTrue(result.format == source.format.label)
                val segments = NativeVad.segments(model, result, model.defaults)
                Log.i(TAG, "${file.name} ${model.label} ${result.format}: ${result.sourceSampleRate} Hz ${result.sourceChannels} ch " +
                    "${result.durationSec}s, ${segments.size} segments in ${result.elapsedMs} ms")
                assertTrue(result.probs.isNotEmpty())

                val json = JSONObject()
                    .put("sampleRate", result.sourceSampleRate)
                    .put("channels", result.sourceChannels)
                    .put("mime", result.mime)
                    .put("duration", result.durationSec.toDouble())
                    .put("elapsedMs", result.elapsedMs)
                    .put("probs", JSONArray(result.probs.map { it.toDouble() }))
                    .put("segments", JSONArray(segments.map { JSONArray(listOf(it.startSec.toDouble(), it.endSec.toDouble())) }))
                File(dir, "${file.name}.${model.name.lowercase()}.${source.format.name.lowercase()}.json").writeText(json.toString())
            }
        }
    }

    private companion object {
        const val TAG = "NativeVadTest"
    }
}
