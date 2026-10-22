package com.example.vadapp.vad

import android.content.Context

/** Persists the user's segmentation settings per model (e.g. a threshold applied from dev mode). */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("segment_settings", Context.MODE_PRIVATE)

    fun load(type: VadModelType): SegmentSettings {
        val d = type.defaults
        val p = type.name
        return SegmentSettings(
            threshold = prefs.getFloat("$p.threshold", d.threshold),
            minSpeechMs = prefs.getInt("$p.minSpeechMs", d.minSpeechMs),
            minSilenceMs = prefs.getInt("$p.minSilenceMs", d.minSilenceMs),
            prePadMs = prefs.getInt("$p.prePadMs", d.prePadMs),
            postPadMs = prefs.getInt("$p.postPadMs", d.postPadMs),
        )
    }

    fun save(type: VadModelType, s: SegmentSettings) {
        val p = type.name
        prefs.edit()
            .putFloat("$p.threshold", s.threshold)
            .putInt("$p.minSpeechMs", s.minSpeechMs)
            .putInt("$p.minSilenceMs", s.minSilenceMs)
            .putInt("$p.prePadMs", s.prePadMs)
            .putInt("$p.postPadMs", s.postPadMs)
            .apply()
    }
}
