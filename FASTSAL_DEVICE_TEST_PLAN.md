# PLAN: Test FastSal .dlc trên thiết bị (HTP, sm8950, QAIRT SDK 2.51)

> Dành cho coding agent (Gemini CLI). Model đã được convert sang `.dlc` — KHÔNG convert lại.
> Thực hiện tuần tự. Mỗi bước có **Tiêu chí hoàn thành**; chưa đạt thì không làm bước sau.

---

## 0. CONFIG — người dùng điền

```yaml
DLC:              <đường dẫn file .dlc, vd D:/work/fastsal/fastsal_fp16.dlc>
ONNX_REF:         <file .onnx gốc để so sánh — để trống nếu không có>
QAIRT_ROOT:       <thư mục SDK 2.51, vd D:/qairt/2.51.x  hoặc /home/<user>/qairt/2.51.x trong WSL>
TEST_IMAGES_DIR:  <thư mục 3–5 ảnh test>
PREPROCESS:       <vd: RGB, resize W×H, /255, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]>
WORK_DIR:         <thư mục làm việc, vd D:/work/fastsal_devtest>
DEVICE_DIR:       /data/local/tmp/fastsal
```

Giá trị nào còn `<...>`: tự tìm nếu có thể; không chắc (nhất là PREPROCESS) thì DỪNG và hỏi người dùng.

---

## 1. Quy tắc

1. Host là Windows. Tool host `qairt-*` / `snpe-*` chạy trong WSL2 sau `source $QAIRT_ROOT/bin/envsetup.sh` (nếu SDK có bản Windows chạy được thì dùng cũng được).
2. `adb` chạy từ PowerShell. Nếu dùng Git Bash: `export MSYS_NO_PATHCONV=1` trước mọi lệnh adb có đường dẫn `/data/...`.
3. Không sửa file `.dlc` gốc. Không đổi phiên bản SDK — mọi binary và `.so` lấy từ cùng QAIRT 2.51.
4. Trước khi dùng flag nào, kiểm tra bằng `<tool> --help`.
5. Ghi mọi lệnh + output vào `WORK_DIR/logs/`. Kết quả cuối ghi vào `WORK_DIR/REPORT.md`.
6. Lỗi: thử tối đa 2 cách sửa hợp lý, rồi DỪNG, ghi lại và hỏi người dùng.

---

## Bước 1 — Đọc thông tin DLC

```bash
qairt-dlc-info -i $DLC > logs/dlc_info.txt
```
Ghi vào REPORT:
- Tên, shape, dtype, **layout (NHWC/NCHW)** của input và output.
- DLC đã có HTP cache chưa, cho SoC nào (nếu dlc-info hiển thị).
- Model là float (FP16/FP32) hay quantized.

**Hoàn thành khi**: biết chắc tên + shape + layout input/output.

---

## Bước 2 — Xác định Hexagon arch & kiểm tra thiết bị

- `adb devices` → thấy máy.
- `adb shell getprop ro.soc.model` → ghi vào REPORT (mong đợi sm8950; khác thì DỪNG và hỏi).
- `adb shell ls -l /vendor/lib64/libcdsprpc.so` → phải tồn tại.
- Tra tài liệu SDK (`$QAIRT_ROOT/docs`, bảng Supported Snapdragon Devices) và `ls $QAIRT_ROOT/lib/` để xác định sm8950 ứng với `hexagon-v??` nào → ghi `HEXAGON_ARCH` vào REPORT.
  - Không xác định được → DỪNG và hỏi người dùng.

**Hoàn thành khi**: có `HEXAGON_ARCH` và thiết bị có `libcdsprpc.so`.

---

## Bước 3 — Tạo input .raw

Script `scripts/make_raw.py`:
- Đọc ảnh trong `TEST_IMAGES_DIR`, áp `PREPROCESS`.
- Ghi float32 bằng `.tofile()` theo **đúng shape + layout của input trong DLC** (Bước 1) → `data/<tên>.raw`.
- Lưu thêm bản NCHW `data/<tên>_nchw.npy` (để chạy ONNX_REF).
- Sinh `data/input_list.txt`: mỗi dòng một tên file `.raw` (đường dẫn tương đối, vì sẽ chạy trong `DEVICE_DIR`). Nếu model có nhiều input: dạng `tên_input:=file.raw`.
- Assert: số byte mỗi file = tích các chiều input × 4.

**Hoàn thành khi**: có các `.raw` đúng kích thước + `input_list.txt`.

---

## Bước 4 — Push lên thiết bị

Script `scripts/push.ps1`:
```
adb shell mkdir -p DEVICE_DIR
adb push QAIRT_ROOT/bin/aarch64-android/snpe-net-run           DEVICE_DIR/
adb push QAIRT_ROOT/bin/aarch64-android/snpe-platform-validator DEVICE_DIR/
adb push QAIRT_ROOT/lib/aarch64-android/.                       DEVICE_DIR/
adb push QAIRT_ROOT/lib/hexagon-<HEXAGON_ARCH>/unsigned/.       DEVICE_DIR/
adb push DLC                                                    DEVICE_DIR/
adb push WORK_DIR/data/*.raw WORK_DIR/data/input_list.txt       DEVICE_DIR/
adb shell chmod +x DEVICE_DIR/snpe-net-run DEVICE_DIR/snpe-platform-validator
```
Sau đó `adb shell ls -l DEVICE_DIR` → lưu vào log, kiểm tra có `libSNPE.so`, `libSnpeHtpPrepare.so`, `libSnpeHtp<ARCH>Stub.so`, `libSnpeHtp<ARCH>Skel.so`.

**Hoàn thành khi**: đủ file trên thiết bị, Stub và Skel cùng `<ARCH>`.

---

## Bước 5 — Chạy

Script `scripts/run.ps1`. Mỗi lệnh dưới chạy trong `adb shell "..."` với prefix môi trường:
```
cd DEVICE_DIR && export LD_LIBRARY_PATH=$PWD && export ADSP_LIBRARY_PATH="$PWD;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" &&
```
(chú ý escape `$PWD` và `;` đúng cách trong PowerShell — `$PWD` phải được shell Android mở rộng, không phải PowerShell.)

Thứ tự:
1. `./snpe-platform-validator --runtime dsp` → ghi PASS/FAIL.
2. CPU (tham chiếu):
   `./snpe-net-run --container <dlc> --input_list input_list.txt --use_cpu --output_dir out_cpu`
3. DSP/HTP:
   `./snpe-net-run --container <dlc> --input_list input_list.txt --use_dsp --perf_profile burst --output_dir out_dsp`
4. Kéo kết quả: `adb pull DEVICE_DIR/out_cpu WORK_DIR/device_out/` và tương tự `out_dsp`.

**Nếu bước 3 lỗi**, chẩn đoán theo thứ tự và ghi từng kết quả:
| Kiểm tra | Nếu fail → nghi ngờ |
|---|---|
| CPU (bước 2) có chạy? | DLC/convert lỗi, hoặc input sai shape |
| platform-validator PASS? | Sai `HEXAGON_ARCH` Stub/Skel, ADSP_LIBRARY_PATH sai |
| Lỗi chứa "cache"/"incompatible"? | HTP cache trong DLC không khớp SoC/SDK |
| Thiếu symbol / `.so` | LD_LIBRARY_PATH, thiếu `libc++_shared.so` |

Thu log: `adb logcat -d | findstr /i "snpe qnn cdsprpc fastrpc htp"` (PowerShell) → `logs/logcat.txt`.

**Hoàn thành khi**: `--use_dsp` chạy thành công và có `device_out/out_dsp/Result_*/`.

---

## Bước 6 — So sánh kết quả

Script `scripts/compare.py`:
- Đọc `device_out/out_{cpu,dsp}/Result_i/<output>.raw` (float32), reshape theo shape output (Bước 1); chuyển NHWC→NCHW nếu cần. Thứ tự `Result_i` khớp thứ tự dòng trong `input_list.txt`.
- Tham chiếu: nếu có `ONNX_REF` → chạy onnxruntime với `*_nchw.npy`; nếu không → dùng output CPU làm tham chiếu.
- Tính cho mỗi ảnh: max_abs_diff, mean_abs_diff, Pearson CC, có NaN/Inf.
- Lưu ảnh `compare_<tên>.png` ghép ngang: ảnh gốc | ref | cpu | dsp (saliency chuẩn hoá về 0–255).
- In bảng tổng hợp.

Ngưỡng tham khảo: CC(dsp, ref) ≥ 0.99 với FP16; ≥ 0.98 với model quantized. Thấp hơn hoặc có NaN → ghi nhận, KHÔNG tự sửa model; đề xuất chạy lại DSP với `--debug` để dump từng layer và hỏi người dùng.

---

## Bước 7 — Thời gian chạy

```bash
snpe-diagview --input_log device_out/out_dsp/SNPEDiag.log > logs/diag_dsp.txt
```
(tương tự cho CPU). Ghi vào REPORT: thời gian init, inference trung bình. Init lâu bất thường (vài giây) → ghi chú khả năng HTP cache không được dùng.

---

## Deliverables

- [ ] `scripts/`: make_raw.py, push.ps1, run.ps1, compare.py — chạy lại được.
- [ ] `device_out/`, `logs/`, các ảnh `compare_*.png`.
- [ ] `REPORT.md`: thông tin DLC, SoC + HEXAGON_ARCH, danh sách `.so` đã dùng, kết quả validator, bảng accuracy CPU/DSP, thời gian init/inference, lỗi & vấn đề còn lại.
