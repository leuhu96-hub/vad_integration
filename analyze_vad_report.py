#!/usr/bin/env python3
"""Đánh giá file .xlsx xuất từ chế độ Dev của app vad_integration.

Chỉ dùng SỐ ĐẾM THÔ ở sheet "Quét ngưỡng" (cột A..I: Model, Cấu hình, Tập, Chế độ, Ngưỡng, TP, FN, FP, TN),
vì các cột công thức có thể chưa có giá trị đã tính khi mở bằng thư viện. Mọi tỷ lệ được tính lại ở đây.

Cách dùng:
  python analyze_vad_report.py vad_dev_report_XXXX.xlsx --mr-max 0.03 --far-max 0.10 \
      [--select-mr 0.025 --select-far 0.09] [--miss-weight 0.75] [--out ket_qua.md]
"""
import argparse, sys
import numpy as np
import pandas as pd

MODES = {"all": "strict", "nb": "collar", "b": "chỉ bin biên"}
CRITS = ["DCF_MIN", "F1_MAX", "YOUDEN", "EER", "MR_MAX", "MR_FAR"]


def load(path):
    df = pd.read_excel(path, sheet_name="Quét ngưỡng", header=1, usecols="A:I")
    df.columns = ["model", "config", "set", "mode", "thr", "TP", "FN", "FP", "TN"]
    df = df.dropna(subset=["TP"]).copy()
    df["thr"] = df["thr"].astype(float).round(2)
    for c in ("TP", "FN", "FP", "TN"):
        df[c] = df[c].astype(float)
    return df


def rates(d):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = d.copy()
        d["MR"] = d.FN / (d.TP + d.FN)
        d["FAR"] = d.FP / (d.FP + d.TN)
        d["F1"] = 2 * d.TP / (2 * d.TP + d.FN + d.FP)
        d["Recall"] = d.TP / (d.TP + d.FN)
    return d


def cost(d, crit, w, x, y):
    if crit == "DCF_MIN": return w * d.MR + (1 - w) * d.FAR
    if crit == "F1_MAX": return -d.F1
    if crit == "YOUDEN": return -(d.Recall - d.FAR)
    if crit == "EER": return (d.MR - d.FAR).abs()
    if crit == "MR_MAX": return d.FAR.where(d.MR <= x + 1e-12)
    if crit == "MR_FAR":
        ok = (d.MR <= x + 1e-12) & (d.FAR <= y + 1e-12)
        return np.maximum(d.MR / x, d.FAR / y).where(ok)
    raise ValueError(crit)


def pick(dev_rows, crit, w, x, y):
    """Ngưỡng tốt nhất trên dev (strict). Hoà -> ngưỡng thấp nhất. None nếu không có."""
    c = cost(dev_rows, crit, w, x, y)
    c = c[c.notna()]
    if c.empty: return None
    best = c.min()
    return float(dev_rows.loc[c[c == best].index].thr.min())


def auc(roc):  # roc: DataFrame FAR, Recall
    p = roc.dropna()[["FAR", "Recall"]].values.tolist() + [[0, 0], [1, 1]]
    p.sort()
    return sum((p[i][0] - p[i - 1][0]) * (p[i][1] + p[i - 1][1]) / 2 for i in range(1, len(p)))


def tpr_at_far(d, f):
    s = d[(d.FAR <= f + 1e-12)].Recall.dropna()
    return float(s.max()) if len(s) else 0.0


def pct(v): return "–" if v is None or pd.isna(v) else f"{100 * v:.2f}%"


def verdict(mr, far, lim_mr, lim_far):
    if pd.isna(mr) or pd.isna(far): return "không có dữ liệu"
    ok = mr < lim_mr and far <= lim_far
    return "ĐẠT" if ok else "KHÔNG ĐẠT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--mr-max", type=float, default=0.03, help="Mục tiêu MR (so sánh kết quả test)")
    ap.add_argument("--far-max", type=float, default=0.10, help="Mục tiêu FAR")
    ap.add_argument("--select-mr", type=float, default=None, help="x của tiêu chí MR_FAR/MR_MAX (mặc định = 0.025)")
    ap.add_argument("--select-far", type=float, default=None, help="y của tiêu chí MR_FAR (mặc định = 0.09)")
    ap.add_argument("--miss-weight", type=float, default=0.75)
    ap.add_argument("--criterion", default="MR_FAR", choices=CRITS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sx = a.select_mr if a.select_mr else 0.025
    sy = a.select_far if a.select_far else 0.09

    df = rates(load(a.xlsx))
    out = []
    P = out.append
    P(f"# Đánh giá {a.xlsx}\n")
    P(f"Mục tiêu: MR < {pct(a.mr_max)}, FAR ≤ {pct(a.far_max)}. Chọn ngưỡng trên dev (strict) bằng {a.criterion}"
      f" (x={pct(sx)}, y={pct(sy)}). Mọi kết luận lấy từ tập TEST.\n")
    configs = sorted(df.config.unique())

    for model in df.model.unique():
        m = df[df.model == model]
        P(f"\n## Model: {model}\n")
        # 1) bảng test theo cấu hình
        P("### 1. Kết quả test (strict) tại ngưỡng chọn trên dev\n")
        P("| Cấu hình | Ngưỡng | MR | FAR | F1 | AUC test | Kết luận |\n|---|---|---|---|---|---|---|")
        chosen = {}
        for cfg in configs:
            dev = m[(m.config == cfg) & (m.set == "dev") & (m["mode"] == "all")]
            tst = m[(m.config == cfg) & (m.set == "test") & (m["mode"] == "all")]
            if dev.empty or tst.empty: continue
            k = pick(dev, a.criterion, a.miss_weight, sx, sy)
            if k is None:
                P(f"| {cfg} | – | – | – | – | {auc(tst):.3f} | không có ngưỡng thỏa tiêu chí trên dev |")
                continue
            chosen[cfg] = k
            r = tst[tst.thr == k].iloc[0]
            P(f"| {cfg} | {k:.2f} | {pct(r.MR)} | {pct(r.FAR)} | {r.F1:.3f} | {auc(tst):.3f} | {verdict(r.MR, r.FAR, a.mr_max, a.far_max)} |")

        # 2) dev vs test
        P("\n### 2. Dev so với test (độ lạc quan)\n")
        P("| Cấu hình | MR dev | MR test | Chênh MR | FAR dev | FAR test | Chênh FAR |\n|---|---|---|---|---|---|---|")
        for cfg, k in chosen.items():
            g = lambda s: m[(m.config == cfg) & (m.set == s) & (m["mode"] == "all") & (m.thr == k)].iloc[0]
            d, t = g("dev"), g("test")
            rel = lambda dv, tv: "–" if dv == 0 else f"{100 * (tv - dv) / dv:+.0f}%"
            P(f"| {cfg} | {pct(d.MR)} | {pct(t.MR)} | {rel(d.MR, t.MR)} | {pct(d.FAR)} | {pct(t.FAR)} | {rel(d.FAR, t.FAR)} |")

        # 3) đóng góp từng bước
        P("\n### 3. Đóng góp của từng bước hậu xử lý (test, strict)\n")
        P("| Bước | ΔMR (điểm %) | ΔFAR (điểm %) |\n|---|---|---|")
        order = [c for c in ["A0", "A1", "A2", "A3", "A4", "A5"] if c in chosen]
        for prev, cur in zip(order, order[1:]):
            g = lambda c: m[(m.config == c) & (m.set == "test") & (m["mode"] == "all") & (m.thr == chosen[c])].iloc[0]
            p_, c_ = g(prev), g(cur)
            P(f"| {prev} → {cur} | {100 * (c_.MR - p_.MR):+.2f} | {100 * (c_.FAR - p_.FAR):+.2f} |")

        # 4) ba chế độ đếm bin
        best = "A5" if "A5" in chosen else (order[-1] if order else None)
        if best:
            P(f"\n### 4. Ba chế độ đếm bin, cấu hình {best} (test)\n")
            P("| Chế độ | MR | FAR |\n|---|---|---|")
            for mode, label in MODES.items():
                r = m[(m.config == best) & (m.set == "test") & (m["mode"] == mode) & (m.thr == chosen[best])]
                if len(r): P(f"| {label} | {pct(r.iloc[0].MR)} | {pct(r.iloc[0].FAR)} |")
            P("\nCollar thấp hơn strict nhiều = lỗi chủ yếu sát ranh giới (GT chỉ chính xác 0.5 s).")

            # 5) khả thi
            tst = m[(m.config == best) & (m.set == "test") & (m["mode"] == "all")]
            P(f"\n### 5. Mức khả thi của mục tiêu (cấu hình {best}, test)\n")
            t = tpr_at_far(tst, a.far_max)
            P(f"- TPR cao nhất khi FAR ≤ {pct(a.far_max)}: **{pct(t)}**; cần ≥ {pct(1 - a.mr_max)} để MR < {pct(a.mr_max)}. "
              f"=> {'khả thi' if t >= 1 - a.mr_max else 'KHÔNG khả thi với model/cấu hình này'}.")
            lo = tst[tst.MR < a.mr_max].FAR.min()
            P(f"- FAR nhỏ nhất khi MR < {pct(a.mr_max)} (ngưỡng nào cũng được, quét trên test để chẩn đoán): {pct(lo)}.")

    # 6) tập file
    try:
        sp = pd.read_excel(a.xlsx, sheet_name="Split", header=1)
        P("\n## Dữ liệu\n")
        P(f"Sheet Split có {len(sp)} dòng. Kiểm tra mỗi ô quality × category có cả dev lẫn test.")
    except Exception:
        pass
    P("\n## Ghi chú\n- Script không tính CI95 (cần số liệu từng file). Đọc CI95 ở sheet 'Tổng hợp' và nhóm ở 'Theo nhóm'.")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    sys.exit(main())
