# VAD Video (Android)

Chọn một video/audio bất kỳ → tách audio → chạy VAD native (C++ + ONNX Runtime) → hiển thị
các đoạn speech trên timeline, phát/seek theo từng đoạn.

Models (trong `app/src/main/assets/models/`):

| Model | File | Đầu vào | Bước | Hậu xử lý |
|---|---|---|---|---|
| FireRedVAD | `fireredvad_vad.onnx` + `cmvn.ark` | fbank 80 (Kaldi) + CMVN | 10 ms | port `VadPostprocessor` gốc |
| Silero VAD v6 | `silero_vad.onnx` | 512 mẫu + 64 mẫu context, state | 32 ms | port `get_speech_timestamps` |
| YAMNet | `yamnet.onnx` (tf2onnx) | waveform, frontend nằm trong graph | 0.48 s | max(Speech, Child speech, Conversation, Narration) ≥ ngưỡng, gộp/lọc |

## Pipeline native (`app/src/main/cpp`)

- `audio_decoder.cpp`: NDK `AMediaExtractor` + `AMediaCodec` → PCM (8/16/24/32-bit, float),
  downmix mọi số kênh về mono, giữ phần frame lẻ giữa các buffer.
- `resampler.h`: resampler sinc cửa sổ Kaiser dạng streaming, sample rate bất kỳ → 16 kHz.
- `fbank.h`: fbank tương thích kaldi-native-fbank (povey, preemph 0.97, snip_edges) + đọc `cmvn.ark`.
- `vad_models.cpp`: chạy model theo kiểu streaming (audio được đẩy vào trong lúc giải mã,
  không phải giữ cả file trong RAM). FireRedVAD chia khối 30000 frame như bản gốc.
- `postprocess.cpp`: xác suất → segment. Gọi riêng qua JNI nên đổi ngưỡng không phải chạy lại model.
- `jni_bridge.cpp`: `NativeVad.analyze(...)` / `NativeVad.segment(...)`.

onnxruntime-android không có prefab: Gradle task `extractOrt` giải nén header + `.so` từ AAR
vào `app/build/ort` cho CMake.

## UI (`app/src/main/java/com/example/vadapp`)

- Chọn model (FireRedVAD / Silero / YAMNet), chọn file (`video/*`, `audio/*`).
- ExoPlayer hiển thị video; timeline tổng quan + timeline chi tiết 20 s theo playhead
  (đường xác suất, ngưỡng, các đoạn speech); chạm/kéo để seek.
- Danh sách segment: chạm để seek, nút ▶ để phát đúng đoạn đó (tự dừng ở cuối đoạn).
- Nút đoạn trước/sau, công tắc "Chỉ speech" (tự nhảy qua khoảng lặng).
- Biểu tượng ⚙ mở tham số: ngưỡng, speech tối thiểu, khoảng lặng tối thiểu, nới rộng 2 đầu.
- Kết quả được cache theo (file, model).

## Build

```
./gradlew assembleDebug        # JDK 17+ (ví dụ JBR của Android Studio)
```

ABI: `arm64-v8a`, `x86_64`. minSdk 26.

## Kiểm tra so với bản tham chiếu

`NativeVadTest` (androidTest) chạy cả 3 model trên mọi file trong `filesDir/vadtest/`
và ghi `<file>.<model>.json` (xác suất + segment) để so với Python (kaldi-native-fbank +
onnxruntime + postprocessor gốc). Hướng dẫn chạy nằm trong comment đầu file test.
