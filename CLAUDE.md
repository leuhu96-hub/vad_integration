# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
./gradlew assembleDebug          # JDK 17+ (e.g. Android Studio's JBR); ABIs arm64-v8a, x86_64; minSdk 26
./gradlew installDebug installDebugAndroidTest
adb shell am instrument -w com.example.vadapp.test/androidx.test.runner.AndroidJUnitRunner
```

`./gradlew testDebugUnitTest` runs the JVM tests of the dev-mode evaluation (`EvalTest`; also writes a sample workbook to `app/build/eval-test/`). The other test is the instrumented `NativeVadTest`, which runs every available model file on every media file in `filesDir/vadtest/` and writes `<name>.<model>.<format>.json` (probabilities + segments) for comparison against a Python reference. Setup steps (adb push / run-as) are in the comment at the top of the test. Run one test class with `-e class com.example.vadapp.NativeVadTest` on the `am instrument` command.

## Architecture

App = Kotlin/Compose UI + a single native library (C++17, CMake in `app/src/main/cpp`) that decodes audio, runs a VAD model, and post-processes to speech segments. README.md (Vietnamese) describes the model table and pipeline in detail; note it still mentions only ONNX, whereas the code now supports ONNX **and** TFLite.

- **Native pipeline**: `audio_decoder.cpp` (NDK MediaCodec → mono PCM) → `resampler.h` (→ 16 kHz) → `vad_models.cpp` (streaming runners for FireRedVAD / Silero / YAMNet; audio is pushed while decoding, never fully held in RAM) → probabilities. `postprocess.cpp` turns probabilities into segments and is exposed separately over JNI (`NativeVad.segment`) so threshold changes don't re-run the model. `jni_bridge.cpp` ↔ `vad/NativeVad.kt`.
- **Runtime abstraction**: `inference.h/.cpp` defines `InferenceSession`; `CreateSession(bytes, threads)` sniffs the file content (TFLite "TFL3" id at byte 4, else ONNX protobuf) and picks ONNX Runtime or TFLite (LiteRT). Runners must go through this interface, not call a runtime directly. Tensors are matched by name, falling back to positional order (needed for TF-Hub YAMNet). The Kotlin side mirrors the sniffing in `vad/ModelRegistry.kt` — keep both in sync.
- **Model files**: format is decided by file content, not extension. `ModelRegistry` lists bundled `assets/models/<assetBase>.{onnx,tflite}` plus user-imported files in `filesDir/models/<type>/`. `cmvn.ark` is needed by FireRedVAD.
- **Dependencies' native bits**: onnxruntime-android and LiteRT AARs have no prefab, so the Gradle tasks `extractOrt` / `extractTflite` unpack headers + `.so` into `app/build/{ort,tflite}` for CMake (CMake tasks depend on them). `libonnxruntime4j_jni.so` is excluded from packaging; `libtensorflowlite_jni.so` must stay (it exports the TFLite C API). Models are `noCompress` so they can be read straight from the APK.
- **UI** (`ui/`, `MainViewModel.kt`): ExoPlayer + overview/detail timelines, segment list, parameters sheet; results are cached per (file, model).
- **Dev mode** (`DevViewModel.kt`, `ui/DevScreen.kt`, `vad/eval/`): evaluates models against 0.5 s-bin ground truth from SAF folders (video tree = quality folders, GT tree = category folders, matched by base name), balanced dev/test split, threshold sweep over configs A0–A4 (A0 = raw bin score, A1–A4 = segmenter with merge/drop/pad added cumulatively, A5 = merge/drop/pad from the optional dev grid search in `ParamSearch`), three bin modes (strict / collar / boundary only), and exports an `.xlsx` whose tables are formulas over the raw counts in sheet "Quét ngưỡng" (layout documented in `XlsxReport`). Keep `Metrics.select` tie-breaking and formulas consistent with the workbook formulas. Model outputs are cached in `filesDir/devcache`.
- **tools/**: `silero_inline_if.py` — offline script that constant-folds Silero's `If` nodes for sr=16000 (used to prepare the ONNX/TFLite model); not part of the build.

Model outputs must match the reference implementations (kaldi-native-fbank + original FireRedVAD postprocessor, Silero `get_speech_timestamps`); fbank/postprocess code is deliberately a port, so change it only with a reference comparison via `NativeVadTest`.
