# FireRedVAD TFLite T = 1000: hướng dẫn sử dụng

Tài liệu mô tả cách dùng model `app/src/main/assets/models/fr_t1000_4d.tflite`: chuẩn bị input từ audio, chạy model có trục thời gian cố định, xử lý output thành các đoạn tiếng nói (segment). Có code ví dụ Python (chạy trên PC) và Kotlin (Android / LiteRT), cùng cách dùng trong chính project này.

---

## 1. Tóm tắt

| | |
|---|---|
| File | `fr_t1000_4d.tflite` (2.37 MB, float32) |
| Input | `feat` — `float32 [1, 1000, 80, 1]` — fbank 80 chiều + CMVN, 1000 frame (10 s) |
| Output | `probs` — `float32 [1, 1000, 1]` — xác suất có tiếng nói của từng frame (đã qua sigmoid, 0..1) |
| Frame | dài 25 ms, bước 10 ms; frame `i` có tâm tại `0.0125 + 0.01·i` giây |
| Receptive field | 152 frame trước + 160 frame sau (≈ 1.5 s + 1.6 s) |
| Op dùng | `RESHAPE`, `CONV_2D` (1×1), `PAD`, `DEPTHWISE_CONV_2D`, `ADD`, `LOGISTIC` |
| File phụ bắt buộc | `cmvn.ark` (thống kê chuẩn hoá Kaldi, `assets/models/cmvn.ark`) |
| Tạo bởi | `tools/export_firered_fixed.py --frames 1000 --channel-dim --out fr_t1000_4d.tflite` |

Chiều cuối `1` của input chỉ là "kênh" cho đúng dạng NHWC mà NPU ưa thích. Thứ tự byte trong bộ nhớ **giống hệt** `[1, 1000, 80]`: 1000 hàng × 80 số float, hàng nọ nối hàng kia.

Độ chính xác (đã kiểm chứng): so với model gốc chạy cả file một lần, xác suất lệch tối đa ~2.5·10⁻⁶. Trên audio 40 phút, danh sách segment **trùng hoàn toàn** (219/219) với `FireRedVad.detect` của thư viện `fireredvad`.

---

## 2. Toàn cảnh pipeline

```
 file media ──► giải mã, trộn mono, resample 16 kHz ──► ×32768 (thang int16)
                                                               │
                                                               ▼
                 fbank 80 chiều (25 ms / 10 ms) ──► CMVN ──► feat [N, 80]
                                                               │
                         cắt chunk ≤ 30000 frame (300 s) ◄─────┘
                                                               │
             cửa sổ trượt T = 1000 frame (chồng lấn 152 / 160) │
                                                               ▼
                          TFLite: feat [1,1000,80,1] ──► probs [1,1000,1]
                                                               │
                        ghép phần "sạch" của mỗi cửa sổ ──► probs [N]
                                                               │
     làm mượt ► ngưỡng ► máy trạng thái ► bù trễ ► nới ► tách đoạn dài
                                                               ▼
                                         segments [(start_s, end_s), ...]
```

---

## 3. Input: từ audio tới `feat`

### 3.1 Audio

- **16 kHz, mono.** Nếu nguồn có nhiều kênh thì lấy trung bình các kênh. Nếu tần số lấy mẫu khác thì resample về 16 kHz.
- **Thang int16.** Fbank kiểu Kaldi tính trên giá trị mẫu trong khoảng `[-32768, 32767]`. Nếu bạn có float trong `[-1, 1]` thì **nhân 32768** trước khi tính fbank. Quên bước này thì log-mel lệch một hằng số lớn và model ra kết quả sai hoàn toàn.

### 3.2 Fbank (khớp kaldi-native-fbank)

| Tham số | Giá trị |
|---|---|
| Tần số lấy mẫu | 16000 Hz |
| Độ dài frame | 25 ms = 400 mẫu |
| Bước frame | 10 ms = 160 mẫu |
| `snip_edges` | true — chỉ lấy frame nằm trọn trong audio |
| Dither | 0 |
| Window | povey |
| Pre-emphasis | 0.97 |
| Bỏ DC (trừ trung bình từng frame) | có |
| Số mel bin | 80 |
| Dải mel | 20 Hz → 8000 Hz (Nyquist) |
| Đầu ra | log năng lượng mel (power spectrum) |

Số frame với `n` mẫu:

```
N = 0                          nếu n < 400
N = 1 + (n − 400) // 160       nếu n ≥ 400
```

Ví dụ: 1 s = 16000 mẫu → 98 frame; 10 phút = 9 600 000 mẫu → 59 998 frame.

### 3.3 CMVN

`cmvn.ark` là ma trận Kaldi kích thước 2 × 81:

- hàng 0: tổng của từng chiều (80 cột), cột cuối = số frame `count`;
- hàng 1: tổng bình phương của từng chiều.

```
mean[d] = sum[d] / count
var[d]  = max(sumsq[d] / count − mean[d]², 1e-20)
istd[d] = 1 / sqrt(var[d])

feat[t][d] = (fbank[t][d] − mean[d]) · istd[d]
```

Sau CMVN, giá trị 0 tương ứng với "mức trung bình" của dữ liệu huấn luyện.

### 3.4 Đưa vào model

`feat` của cả file có dạng `[N, 80]`. Model chỉ nhận đúng **1000 frame** mỗi lần, nên phải chia cửa sổ như ở mục 4. Mỗi lần gọi, chép 1000 × 80 số float liên tiếp (frame `start` … `start+999`) vào tensor input `[1, 1000, 80, 1]`.

---

## 4. Chạy model khi T cố định

### 4.1 Vì sao không cắt thẳng thành các khúc 1000 frame?

Model là DFSMN gồm 8 lớp FSMN. Mỗi lớp "nhìn" 19 frame quá khứ và 20 frame tương lai, nên output tại frame `t` phụ thuộc vào input trong khoảng `[t − 152, t + 160]`. Nếu cắt thẳng thành các khúc 1000 frame rời nhau, các frame gần biên khúc sẽ thiếu ngữ cảnh. Model khi đó coi phần ngoài khúc là 0, và xác suất bị sai trong khoảng 1.5 s quanh mỗi biên.

### 4.2 Cửa sổ trượt chồng lấn

Mỗi cửa sổ chỉ **giữ** những frame có đủ 152 frame phía trước và 160 frame phía sau nằm trong cửa sổ, hoặc nằm sát biên thật của chunk (ở đó model gốc cũng không có ngữ cảnh, nên kết quả vẫn trùng):

```
chunk N frame:   |0 ─────────────────────────────────────────────────────── N|
cửa sổ 1:        [██████ giữ 0..839 ██████|░ 160 ░]
cửa sổ 2:                        [░ 152 ░|█ giữ 840..1527 █|░ 160 ░]
cửa sổ 3:                                          [░ 152 ░|█ giữ ... █|░ 160 ░]
  ...
cửa sổ cuối:                                         [░ 152 ░|██ giữ ... N−1 ██]   start = N − 1000
```

Thuật toán, với `done` là frame đầu tiên chưa có xác suất:

```
start = min(max(0, done − 152), N − 1000)
end   = N                       nếu start + 1000 == N   (cửa sổ chạm cuối chunk)
        start + 1000 − 160      nếu không
giữ probs_cửa_sổ[done − start : end − start]  →  probs[done : end]
done  = end
```

- Mỗi lần gọi model cho ra 688 frame mới (1000 − 152 − 160), riêng cửa sổ đầu và cuối cho nhiều hơn. File 10 phút cần khoảng 88 lần gọi, tức tính thừa khoảng 1.45 lần.
- **Chunk ≤ 1000 frame** (file dưới 10 s, hoặc phần dư sau mốc 300 s): chép vào cửa sổ, **pad 0** phần còn thiếu, chạy một lần rồi lấy `N` giá trị đầu. Pad 0 không tương đương cách model gốc xử lý biên, nên khoảng 150 frame cuối lệch nhẹ (đo được tối đa ~0.015 trên file 7 s, ~0.07 trong trường hợp xấu). Đây là giới hạn không tránh được của shape cố định.

### 4.3 Chunk 300 s

Thư viện gốc tự cắt `feat` thành các chunk 30000 frame và chạy từng chunk độc lập. Muốn khớp tuyệt đối thì làm y như vậy: chia `feat` thành các chunk 30000 frame, chạy cửa sổ trượt **bên trong từng chunk**, rồi nối kết quả. Code native của app (`FireRedVadModel::Feed` / `RunFixed`) và các ví dụ dưới đây đều làm theo cách này.

---

## 5. Output: ý nghĩa của `probs`

- `probs[i]` ∈ [0, 1] là xác suất có tiếng nói của frame `i`. Model đã áp sigmoid nên **không** cần softmax hay sigmoid nữa.
- Thời điểm của frame: tâm tại `0.0125 + 0.01·i` s. Frame phủ đoạn `[0.01·i, 0.01·i + 0.025)` s.
- Số giá trị bằng đúng số frame `N` của file (mục 3.2).
- Xác suất thô dao động nhanh. Đừng ngưỡng hoá trực tiếp; hãy dùng bộ hậu xử lý ở mục 6.

Ví dụ một đoạn `probs` (30 frame = 0.3 s):

```
idx    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
prob .02 .03 .05 .10 .60 .80 .90 .95 .92 .30 .85 .90 .93 .91 .88 .86 .90 .92 .40 .89 .91 .94 .20 .05 .03 .70 .10 .04 .02 .02
```

---

## 6. Hậu xử lý: `probs` → segments

Đây là bản port của `VadPostprocessor` trong FireRedVAD (`app/src/main/cpp/postprocess.cpp`). Tham số mặc định của app (`VadModelType.FIRERED`), quy ra frame 10 ms:

| Tham số | Mặc định | Frame | Ý nghĩa |
|---|---|---|---|
| `smooth` | — | 5 | cửa sổ trung bình trượt |
| `threshold` | 0.4 | — | ngưỡng speech |
| `min_speech` | 200 ms | 20 | đoạn nói ngắn hơn thì bỏ |
| `min_silence` | 200 ms | 20 | khoảng lặng ngắn hơn thì lấp |
| `max_speech` | 20 s | 2000 | đoạn dài hơn thì tách |
| `pre_pad` / `post_pad` | 0 ms | 0 | nới đầu / cuối mỗi đoạn |

Các bước theo đúng thứ tự:

1. **Làm mượt.** Trung bình trượt về phía sau của 5 frame; các frame đầu dùng trung bình tích luỹ. Bước này lấp các "lỗ" ngắn và dập các đỉnh lẻ, nhưng làm sườn lên trễ khoảng 5 frame.
2. **Ngưỡng.** `binary[t] = smoothed[t] ≥ threshold`.
3. **Máy trạng thái** với 4 trạng thái `SILENCE → POSSIBLE_SPEECH → SPEECH → POSSIBLE_SILENCE`:
   - SILENCE gặp 1 thì chuyển sang POSSIBLE_SPEECH và ghi nhớ điểm bắt đầu.
   - POSSIBLE_SPEECH: nếu 1 liên tục đủ `min_speech` frame thì chuyển sang SPEECH và **tô lại** từ điểm bắt đầu. Gặp 0 trước đó thì về SILENCE.
   - SPEECH gặp 0 thì chuyển sang POSSIBLE_SILENCE (vẫn tính là speech, gọi là "hangover").
   - POSSIBLE_SILENCE: nếu 0 liên tục đủ `min_silence` frame thì về SILENCE; gặp 1 thì quay lại SPEECH.
4. **Bù trễ của bước làm mượt.** Ở mỗi sườn 0→1, tô thêm `smooth` frame phía trước.
5. **Nới đoạn** theo `pre_pad` / `post_pad`.
6. **Tách đoạn dài.** Với đoạn dài hơn `max_speech`, tìm frame có **xác suất thô** nhỏ nhất trong nửa sau của mỗi cửa sổ `max_speech` và đặt nó về 0. Lặp lại tới khi không còn đoạn nào quá dài.
7. **Đổi sang giây.** Mỗi chuỗi số 1 liên tiếp `[s, e)` thành `(0.01·s, 0.01·e)`. Đoạn chạm cuối file có `end = min(0.01·N + 0.025, độ dài file)`. Làm tròn tới ms.

Minh hoạ trên 30 frame ở mục 5 (tham số thu nhỏ để dễ nhìn: threshold 0.5, min_speech = min_silence = 3, pad 1, max_speech 12):

```
smooth  .02 .03 .03 .05 .16 .32 .49 .67 .83 .77 .78 .78 .78 .78 .89 .90 .90 .89 .79 .79 .80 .81 .67 .60 .43 .38 .22 .18 .18 .18
binary    0   0   0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0   0   0   0
state     0   0   0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0
fix       0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0
extend    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0
split     0   1   1   1   1   1   1   1   1   0   1   1   1   1   1   1   1   1   0   1   1   1   1   1   1   1   1   1   0   0
→ segments: (0.01, 0.09) (0.10, 0.18) (0.19, 0.28)
```

Giải thích từng dòng: "lỗ" ở frame 9 và 18 được làm mượt lấp đi. Đỉnh lẻ ở frame 25 không tạo ra đoạn nào. Điểm bắt đầu lùi từ 7 về 2 nhờ bước bù trễ. Đoạn 27 frame dài hơn 12 nên bị tách tại frame 9 và 18, là những chỗ xác suất thô thấp nhất.

> Đổi ngưỡng hay tham số chỉ cần chạy lại bước 6 trên `probs` đã có; không phải chạy lại model.

---

## 7. Code ví dụ Python (PC)

Cài đặt (Linux / macOS / Colab; trên Windows có thể LiteRT bị Smart App Control chặn):

```
pip install numpy soundfile kaldi-native-fbank kaldiio ai-edge-litert
```

File `firered_tflite_example.py`. Đã kiểm chứng: trên 40 phút audio, kết quả cho ra danh sách segment trùng với `FireRedVad.detect`.

```python
"""FireRedVAD TFLite T=1000: audio -> feat -> probs -> speech segments (end-to-end example)."""
import math

import kaldi_native_fbank as knf
import kaldiio
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
NUM_BINS = 80
CHUNK_FRAMES = 30000           # reference chunk_max_frame (300 s)
LOOKBACK, LOOKAHEAD = 152, 160  # DFSMN receptive field (frames)


# ---------------------------------------------------------------- 1. audio
def load_audio_int16(path):
    """Mono 16 kHz samples on the int16 scale (float64), like the reference."""
    wav, sr = sf.read(path, dtype="int16", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: {sr} Hz, resample to 16 kHz first")
    return wav.mean(axis=1)  # downmix; values stay in [-32768, 32767]


# ---------------------------------------------------------------- 2. fbank + CMVN
def fbank(wav_int16):
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = NUM_BINS
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, wav_int16.tolist())
    if fb.num_frames_ready == 0:
        return np.zeros((0, NUM_BINS), np.float32)
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)


def load_cmvn(path):
    stats = kaldiio.load_mat(path)          # [2, 81]: sums, sums of squares, count in the last column
    count = stats[0, -1]
    mean = stats[0, :-1] / count
    var = np.maximum(stats[1, :-1] / count - mean * mean, 1e-20)
    return mean.astype(np.float32), (1.0 / np.sqrt(var)).astype(np.float32)


def make_feat(wav_int16, cmvn):
    mean, istd = cmvn
    return (fbank(wav_int16) - mean) * istd  # [N, 80]


# ---------------------------------------------------------------- 3. model (fixed T)
class FixedTModel:
    """Wraps the TFLite model: feat [1, T, 80] or [1, T, 80, 1] -> probs [1, T, 1]."""

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter  # or tf.lite.Interpreter
        self.interp = Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.shape = tuple(self.inp["shape"])  # (1, 1000, 80) or (1, 1000, 80, 1)
        self.frames = self.shape[1]

    def __call__(self, window):             # window: [T, 80] float32
        self.interp.set_tensor(self.inp["index"], window.reshape(self.shape))
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"]).reshape(-1)  # [T]


def run_chunk(model, feat):
    """Probabilities for one chunk (<= 30000 frames) with overlapping windows."""
    t, n = model.frames, len(feat)
    left, right = LOOKBACK, LOOKAHEAD
    if t <= left + right:                    # too short for the receptive field (approximate)
        left = right = t // 4
    if n <= t:                               # short chunk: zero padding (last frames slightly off)
        window = np.zeros((t, NUM_BINS), np.float32)
        window[:n] = feat
        return model(window)[:n]
    probs, done = [], 0
    while done < n:
        start = min(max(0, done - left), n - t)
        end = n if start + t == n else start + t - right
        probs.append(model(np.ascontiguousarray(feat[start:start + t]))[done - start:end - start])
        done = end
    return np.concatenate(probs)


def run_model(model, feat):
    """Same 300 s chunking as the reference, so results match it everywhere."""
    if len(feat) == 0:
        return np.zeros(0, np.float32)
    return np.concatenate([run_chunk(model, feat[i:i + CHUNK_FRAMES])
                           for i in range(0, len(feat), CHUNK_FRAMES)])


# ---------------------------------------------------------------- 4. probs -> segments
def postprocess(probs, wav_dur, threshold=0.4, smooth=5, min_speech=20, min_silence=20,
                max_speech=2000, extend_pre=0, extend_post=0):
    """Port of FireRedVAD VadPostprocessor (frames are 10 ms). Returns [(start_s, end_s)]."""
    n = len(probs)
    if n == 0:
        return []
    # (1) trailing moving average, cumulative mean for the first frames
    csum = np.cumsum(np.concatenate([[0.0], probs]))
    idx = np.arange(n)
    lo = np.maximum(0, idx + 1 - smooth)
    smoothed = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    # (2) threshold
    binary = smoothed >= threshold
    # (3) state machine: min_speech / min_silence
    d = np.zeros(n, np.int8)
    state, s_start, sil_start = "sil", -1, -1
    for t in range(n):
        if state == "sil":
            if binary[t]:
                state, s_start = "maybe_speech", t
        elif state == "maybe_speech":
            if binary[t]:
                if t - s_start >= min_speech:
                    state = "speech"
                    d[s_start:t] = 1
            else:
                state = "sil"
        elif state == "speech":
            if not binary[t]:
                state, sil_start = "maybe_sil", t
        else:  # maybe_sil
            if not binary[t]:
                if t - sil_start >= min_silence:
                    state = "sil"
            else:
                state = "speech"
        d[t] = state in ("speech", "maybe_sil")
    # (4) undo the smoothing delay: every 0->1 edge starts `smooth` frames earlier
    fixed = d.copy()
    for t in range(1, n):
        if d[t - 1] == 0 and d[t] == 1:
            fixed[max(0, t - smooth):t] = 1
    d = fixed
    # (5) pad segments
    if extend_pre or extend_post:
        ext = np.zeros_like(d)
        for t in np.flatnonzero(d):
            ext[max(0, t - extend_pre):min(n, t + extend_post + 1)] = 1
        d = ext
    # (6) split segments longer than max_speech at the lowest raw probability
    out = d.copy()
    for sf_, ef in runs(d):
        start = 0
        while (ef - sf_) - start > max_speech:
            ws, we = start + max_speech // 2, start + max_speech
            best = ws + int(np.argmin(probs[sf_ + ws:sf_ + we]))
            out[sf_ + best] = 0
            start = best + 1
    # (7) frames -> seconds
    segs = []
    for s, e in runs(out):
        end = e * 0.01 if e < n else min(n * 0.01 + 0.025, wav_dur)
        segs.append((round(float(s) * 0.01, 3), round(float(end), 3)))
    return segs


def runs(d):
    """[start, end) frame ranges of consecutive ones."""
    edges = np.diff(np.concatenate([[0], d, [0]]).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# ---------------------------------------------------------------- 5. all together
def detect(wav_path, model_path, cmvn_path, **params):
    wav = load_audio_int16(wav_path)
    feat = make_feat(wav, load_cmvn(cmvn_path))
    probs = run_model(FixedTModel(model_path), feat)
    # probs[i] describes the audio centred at 0.0125 + 0.01 * i seconds
    return postprocess(probs, len(wav) / SAMPLE_RATE, **params), probs


if __name__ == "__main__":
    import sys
    segments, probs = detect(sys.argv[1], "fr_t1000_4d.tflite", "cmvn.ark")
    print(f"{len(probs)} frames, {len(segments)} segments")
    for s, e in segments:
        print(f"{s:8.3f} - {e:8.3f} s")
```

Chạy:

```
python firered_tflite_example.py clip_16k.wav
# 242438 frames, 219 segments
#   10.120 -   13.380 s
#   23.010 -   24.180 s
#   ...
```

Muốn đổi tham số thì truyền thêm vào hàm, ví dụ `detect(path, model, cmvn, threshold=0.5, extend_pre=10, extend_post=10)`; giá trị tính bằng frame 10 ms.

---

## 8. Code ví dụ Kotlin (Android, LiteRT)

Dành cho trường hợp tự gọi model từ Kotlin/Java, ví dụ khi ghép với một runtime NPU có API Java. Phần fbank + CMVN giả định đã có sẵn (trong project này do code native làm). Lớp dưới đây chỉ lo phần cửa sổ trượt và gọi model. Đã kiểm tra biên dịch được với dependency `com.google.ai.edge.litert:litert` của project.

```kotlin
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** FireRedVAD TFLite with a fixed time axis: feat [1,T,80] or [1,T,80,1] -> probs [1,T,1]. */
class FireRedFixedT(modelFile: File, numThreads: Int = 4) : Closeable {
    private val interpreter = Interpreter(modelFile, Interpreter.Options().setNumThreads(numThreads))
    val frames: Int = interpreter.getInputTensor(0).shape()[1]          // 1000
    private val input = ByteBuffer.allocateDirect(frames * BINS * 4).order(ByteOrder.nativeOrder())
    private val output = ByteBuffer.allocateDirect(frames * 4).order(ByteOrder.nativeOrder())

    /** [feat] = fbank + CMVN of one chunk (<= 30000 frames), row-major N x 80. Returns N probabilities. */
    fun run(feat: FloatArray): FloatArray {
        val n = feat.size / BINS
        val probs = FloatArray(n)
        var left = LOOKBACK
        var right = LOOKAHEAD
        if (frames <= left + right) { left = frames / 4; right = frames / 4 }
        if (n <= frames) {                         // short chunk: zero-padded, last frames slightly off
            infer(feat, 0, n).copyInto(probs, 0, 0, n)
            return probs
        }
        var done = 0
        while (done < n) {
            val start = minOf(maxOf(0, done - left), n - frames)
            val end = if (start + frames == n) n else start + frames - right
            infer(feat, start, frames).copyInto(probs, done, done - start, end - start)
            done = end
        }
        return probs
    }

    /** Runs one window: frames [start, start + count) of [feat], zero-padded up to T. */
    private fun infer(feat: FloatArray, start: Int, count: Int): FloatArray {
        input.clear()
        val fb = input.asFloatBuffer()
        fb.put(feat, start * BINS, count * BINS)
        while (fb.hasRemaining()) fb.put(0f)
        output.clear()
        interpreter.run(input, output)
        output.rewind()
        return FloatArray(frames).also { output.asFloatBuffer().get(it) }
    }

    override fun close() = interpreter.close()

    companion object {
        const val BINS = 80
        const val LOOKBACK = 152   // DFSMN receptive field, frames before t
        const val LOOKAHEAD = 160  // frames after t
    }
}
```

Cách dùng với file dài: chia `feat` thành các chunk 30000 frame, gọi `run` cho từng chunk rồi nối kết quả:

```kotlin
FireRedFixedT(modelFile).use { model ->
    val chunk = 30_000 * FireRedFixedT.BINS
    val probs = (0 until feat.size step chunk)
        .map { model.run(feat.copyOfRange(it, minOf(it + chunk, feat.size))) }
        .reduce { a, b -> a + b }
    // probs[i] ↔ thời điểm 0.0125 + 0.01 * i giây → hậu xử lý như mục 6
}
```

---

## 9. Dùng trong project này

Code native đã hỗ trợ sẵn T cố định (`FireRedVadModel::RunFixed` trong `app/src/main/cpp/vad_models.cpp`). Khi model có `feat` với T > 0, code tự chạy cửa sổ trượt như mục 4; không cần sửa gì thêm.

### 9.1 Cho app thấy model

`ModelRegistry` chỉ liệt kê asset tên `fireredvad_vad.onnx` / `fireredvad_vad.tflite` và các file người dùng nạp vào. File `fr_t1000_4d.tflite` hiện nằm trong assets dưới tên khác nên **không hiện trong danh sách**. Chọn một trong các cách:

1. **Nạp qua app:** màn chính → chọn FireRedVAD → nút nạp model → chọn file `.tflite`. File được chép vào `filesDir/models/...` và hiện bên cạnh các bản có sẵn.
2. **Đổi tên asset** thành `fireredvad_vad.tflite`. Cách này thay bản TFLite T động đang có.
3. **Gọi thẳng bằng code** với đường dẫn asset, không cần qua `ModelRegistry`:

```kotlin
val afd = context.contentResolver.openAssetFileDescriptor(videoUri, "r")!!
val result = afd.use {
    NativeVad.analyze(
        context.assets,
        VadModelType.FIRERED.id,
        "asset:models/fr_t1000_4d.tflite",   // model T = 1000
        VadModelType.FIRERED.extraAsset,       // "asset:models/cmvn.ark"
        it.parcelFileDescriptor.fd, it.startOffset, it.length,
        /* numThreads = */ 4,
    ) { fraction -> true }                      // trả false để huỷ
}
// result.probs: 1 giá trị / 10 ms; result.hopSec = 0.01; result.offsetSec = 0.0125
val segments = NativeVad.segments(VadModelType.FIRERED, result, VadModelType.FIRERED.defaults)
segments.forEach { println("${it.startSec} - ${it.endSec} s") }
```

`VadModelType.FIRERED.defaults` là `SegmentSettings(threshold = 0.4, minSpeechMs = 200, minSilenceMs = 200, prePadMs = 0, postPadMs = 0)`. Muốn đổi thì dùng `.copy(threshold = 0.5f)`, v.v. `max_speech` (20 s) và cửa sổ làm mượt (5 frame) cố định trong native.

Hai bước `analyze` và `segments` tách riêng: `analyze` giải mã audio và chạy model (chậm), còn `segments` chỉ chạy hậu xử lý trên `probs` (nhanh). Đổi ngưỡng thì chỉ cần gọi lại `segments`.

### 9.2 Kiểm chứng trên thiết bị

`NativeVadTest` chạy mọi model mà `ModelRegistry` thấy, trên mọi file trong `filesDir/vadtest/`, rồi ghi `<tên>.<model>.<format>.json` (probs + segments):

```
adb push clip.mp4 /data/local/tmp/
adb shell run-as com.example.vadapp sh -c 'mkdir -p files/vadtest && cp /data/local/tmp/clip.mp4 files/vadtest/'
./gradlew installDebug installDebugAndroidTest
adb shell am instrument -w -e class com.example.vadapp.NativeVadTest com.example.vadapp.test/androidx.test.runner.AndroidJUnitRunner
```

Nạp model T = 1000 vào app trước (mục 9.1, cách 1). Sau đó so file JSON của bản T = 1000 với bản ONNX: probs phải lệch dưới khoảng 1e-5 (trừ đoạn cuối của file ngắn hơn 10 s), và segments phải trùng nhau.

---

## 10. Sai sót thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Xác suất gần như không đổi, toàn 0 hoặc toàn 1 | Quên nhân 32768 trước fbank, hoặc quên CMVN | Xem mục 3.1 và 3.3 |
| Xác suất nhảy bậc mỗi 10 s | Cắt khúc 1000 frame rời nhau, không chồng lấn | Dùng cửa sổ trượt (mục 4.2) |
| Segment lệch thời gian khoảng 12 ms | Dùng `0.01·i` thay vì tâm frame `0.0125 + 0.01·i` khi vẽ probs | Segment tính theo `0.01·i` là đúng chuẩn gốc; tâm frame chỉ dùng khi vẽ |
| Thiếu đoạn đầu file | Không bù trễ làm mượt (bước 4, mục 6) | Áp `FixSmoothWindowStart` |
| Clip dưới 10 s lệch nhẹ ở cuối | Pad 0 khi chunk ngắn hơn T (mục 4.2) | Giới hạn của shape cố định; nếu cần, xuất model T nhỏ hơn (≥ 313) |
| `cannot resize input` / sai shape | Gửi `[1, N, 80]` với N ≠ 1000 | Luôn gửi đúng shape khai báo `[1, 1000, 80, 1]` |
| Model mới có kiến trúc khác (AED, ...) | Receptive field khác 152/160 | Chạy `tools/export_firered_fixed.py` (in ra receptive field), sửa `kLookback` / `kLookahead` |

---

## 11. Tạo lại model với T khác

```
python tools/export_firered_fixed.py --ckpt <FireRedVAD>/VAD/model.pth.tar \
    --frames 1000 --channel-dim --out fr_t1000_4d.tflite
```

- `--frames T`: chọn T > 312 để kết quả trùng model gốc. T càng lớn thì càng ít tính thừa (T = 400 thừa khoảng 4.5 lần, T = 1000 khoảng 1.45 lần, T = 2000 khoảng 1.2 lần), nhưng tensor lớn hơn.
- Bỏ `--channel-dim` nếu muốn input `[1, T, 80]`.
- Thêm `x.onnx` vào `--out` để xuất cả ONNX.
- Không cần TensorFlow. Nếu import được `ai-edge-litert`, script tự so TFLite với PyTorch và in ra sai lệch.
