# Nhiệm vụ cho Gemini: đánh giá kết quả VAD từ file xuất của app

Bạn được cung cấp:
- `vad_dev_report_*.xlsx`: file xuất từ chế độ Dev của app vad_integration (thật, do người dùng cung cấp).
- `analyze_vad_report.py`: script tính sẵn các bảng từ số đếm thô.
- (Tuỳ chọn) ảnh ROC `roc_*_all.png`.

Mục tiêu của người dùng: **MR < 3% và FAR ≤ 10%** (ưu tiên không bỏ sót lời nói), đo trên tập **test**, chế độ **strict**.

## Quy tắc bắt buộc
1. Không bịa số. Mọi con số trong báo cáo phải đến từ file xlsx hoặc từ output của script. Không có thì ghi "không có dữ liệu".
2. Kết luận chỉ dựa vào tập **test**. Số dev chỉ để so sánh độ lạc quan.
3. Không sửa file xlsx gốc. Làm việc trên bản sao.
4. Khi đọc xlsx bằng thư viện, các cột công thức có thể trống. Chỉ tin số đếm thô ở sheet "Quét ngưỡng" (cột A đến I: Model, Cấu hình, Tập, Chế độ, Ngưỡng, TP, FN, FP, TN). Script đã làm đúng việc này.
5. Nếu số file test ít hoặc một nhóm có dưới 5 file, ghi rõ độ tin cậy thấp.

## Các bước
1. Chạy:
   `python analyze_vad_report.py vad_dev_report_XXXX.xlsx --mr-max 0.03 --far-max 0.10 --select-mr 0.025 --select-far 0.09 --out ket_qua.md`
   (đổi `--select-mr/--select-far` nếu người dùng đã dùng x, y khác ở mục 4 của app; kiểm tra ô B11 và B17 sheet "Tổng hợp").
2. Đọc sheet **"Tổng hợp"** để lấy các số mà script không tính: CI95 bootstrap (chữ xanh) của MR, FAR, F1, DCF, và tham số đã dùng (ô nền vàng).
3. Đọc sheet **"Tìm tham số"** (nếu có): bảng top 10 tổ hợp và ảnh hưởng từng tham số.
4. Đọc sheet **"Theo nhóm"**: tìm nhóm quality/category có MR hoặc FAR cao hơn trung bình ít nhất 1,5 lần, ghi số file của nhóm.
5. Đọc sheet **"Split"**: kiểm tra mỗi ô quality × category có cả dev lẫn test, và tỷ lệ speech dev/test gần nhau.
6. Viết báo cáo theo mẫu dưới đây.

## Quy tắc kết luận
- **Đạt có cơ sở:** MR và FAR trung bình trong giới hạn **và** cận trên CI95 cũng trong giới hạn.
- **Có thể đạt, chưa chắc:** trung bình trong giới hạn nhưng cận trên CI95 vượt.
- **Không đạt:** trung bình đã vượt giới hạn.
- Dev → test: nếu MR hoặc FAR test cao hơn dev hơn 50% (tương đối), nghi khớp quá vào dev.
- Bước hậu xử lý có ΔMR và ΔFAR đều dưới 0,2 điểm % được coi là không đáng kể.
- Collar thấp hơn strict nhiều ⇒ lỗi chủ yếu sát ranh giới 0.5 s; collar vẫn cao ⇒ lỗi thật giữa vùng.
- Khả thi: TPR@FAR ≤ 10% nhỏ hơn 97% ⇒ mục tiêu không đạt được với model/cấu hình này, dù hậu xử lý thế nào.
- So sánh model: chênh lệch nhỏ hơn bề rộng CI95 ⇒ coi như hoà.

## Mẫu báo cáo (Markdown, tiếng Việt)

```
# Báo cáo đánh giá VAD – <ngày>

## 1. Kết luận ngắn
<Đạt có cơ sở / Có thể đạt, chưa chắc / Không đạt>. <1-2 câu: MR, FAR test và CI95 của cấu hình tốt nhất>.

## 2. Cấu hình đã dùng
Model, định dạng, tiêu chí chọn, x, y, tỷ lệ bin, lưới, bộ A5 (ngưỡng, merge, drop, pre/post pad, làm mượt).

## 3. Kết quả test
Bảng A0..A5 (MR, FAR, F1, AUC, Kết luận) + CI95 của cấu hình tốt nhất.

## 4. Phát hiện
Mỗi phát hiện: Quan sát (số + nguồn: sheet/bảng) → Kết luận → Độ tin cậy → Hành động.
- Đóng góp từng bước hậu xử lý (merge, drop, pad)
- Tham số nào ảnh hưởng nhiều nhất (bảng ảnh hưởng ở sheet "Tìm tham số")
- Dev so với test
- Collar so với strict
- Nhóm yếu

## 5. Hạn chế
GT 0.5 s, số file test, CI95 rộng, lạc quan do chọn trên dev, các điều chưa kiểm tra.

## 6. Việc tiếp theo
Danh sách ngắn, ưu tiên theo tác động.
```

## Việc cần làm khi không tổ hợp nào thỏa x, y
Ghi trong báo cáo: (a) FAR nhỏ nhất đạt được khi MR < 3% (script mục 5), (b) TPR@FAR 10%, (c) đề xuất nới lưới (merge, pad lớn hơn), hạ "Speech trong bin ≥", rồi nới y trước x. Không tự đổi mục tiêu của người dùng.
