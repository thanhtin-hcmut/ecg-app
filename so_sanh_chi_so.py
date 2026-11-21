"""
So sánh các chỉ số ECG với khoảng tham chiếu đã công bố, và giải thích mức lệch theo kiểu SHAP.

Mô hình giải thích (minh bạch, cộng tính):
    Điểm lệch tổng hợp  f(x) = Σ φ_i ,   φ_i = w_i · c_i · min(|d_i|, 2)
      d_i : độ lệch chuẩn hoá của chỉ số i so với khoảng tham chiếu (0 nếu nằm trong khoảng;
            1 = lệch ra ngoài một đoạn bằng độ rộng khoảng)
      c_i : hệ số độ tin cậy của phép đo (Tốt 1,0 · Trung bình 0,7 · Kém 0,4)
      w_i : trọng số, mặc định bằng nhau (1) — không giả định chỉ số nào quan trọng hơn về lâm sàng
    Giá trị nền E[f] = 0 (mọi chỉ số nằm trong khoảng tham chiếu).
Vì f là hàm cộng tính theo từng chỉ số, giá trị Shapley (SHAP) của chỉ số i đúng bằng φ_i,
nên biểu đồ "đóng góp" là SHAP chính xác, không phải xấp xỉ.

Đây KHÔNG phải mô hình chẩn đoán: chỉ cho biết chỉ số nào nằm ngoài khoảng tham chiếu và lệch bao nhiêu.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

HE_SO_TIN_CAY = {"Tốt": 1.0, "Trung bình": 0.7, "Kém": 0.4}
MAU = {"Bình thường": "#2e7d32", "Cao": "#c62828", "Thấp": "#1565c0", "Không đánh giá": "#9e9e9e"}


# ----------------------------------------------------------------------------
# Khoảng tham chiếu
# ----------------------------------------------------------------------------
def khoang_tham_chieu(thai_nhi: bool, gioi: str) -> dict:
    """(cận dưới, cận trên, đơn vị, nguồn). None = không giới hạn phía đó."""
    if thai_nhi:
        return {
            "Nhịp tim thai cơ bản": (110, 160, "l/p", "FIGO 2015"),
            "Biến thiên nhịp tim thai": (5, 25, "l/p", "FIGO 2015 (biên độ dao động trong 1 phút)"),
        }
    qtc_tren = 460 if gioi == "Nữ" else 450
    return {
        "Nhịp tim": (60, 100, "l/p", "AHA – nhịp xoang khi nghỉ"),
        "PR": (120, 200, "ms", "AHA/ACCF/HRS 2009"),
        "QRS": (None, 110, "ms", "AHA/ACCF/HRS 2009 (≥ 110 ms: rối loạn dẫn truyền trong thất)"),
        "QTc (Bazett)": (350, qtc_tren, "ms", f"AHA/ACCF/HRS 2009 ({gioi.lower()} ≤ {qtc_tren} ms)"),
        "SDNN": (32, 93, "ms", "Nunan 2010 – HRV ngắn hạn 5 phút khi nghỉ"),
        "RMSSD": (19, 75, "ms", "Nunan 2010 – HRV ngắn hạn 5 phút khi nghỉ"),
    }


# ----------------------------------------------------------------------------
# Chỉ số riêng cho thai nhi
# ----------------------------------------------------------------------------
def bien_thien_nhip_thai(rpeaks, fs) -> float:
    """Biến thiên nhịp tim thai (FIGO): biên độ dao động của nhịp tim trong từng cửa sổ 1 phút,
    lấy trung vị. Dùng phân vị 5–95% để loại nhịp nhiễu."""
    t = np.asarray(rpeaks[1:]) / fs
    hr = 60 * fs / np.diff(rpeaks)
    if len(t) < 10 or t[-1] - t[0] < 30:
        return np.nan
    bien_do = []
    for a in np.arange(t[0], t[-1], 60):
        h = hr[(t >= a) & (t < a + 60)]
        if len(h) >= 20:
            bien_do.append(np.percentile(h, 95) - np.percentile(h, 5))
    return float(np.median(bien_do)) if bien_do else np.nan


# ----------------------------------------------------------------------------
# Độ tin cậy phép đo
# ----------------------------------------------------------------------------
def _muc(sqi: float, ha_bac: int = 0) -> str:
    ds = ["Tốt", "Trung bình", "Kém"]
    i = 0 if sqi >= 0.85 else 1 if sqi >= 0.7 else 2
    return ds[min(i + ha_bac, 2)]


# ----------------------------------------------------------------------------
# Đánh giá
# ----------------------------------------------------------------------------
def danh_gia_chi_so(gia_tri: dict, thai_nhi: bool, gioi: str, sqi: float, thoi_luong_s: float,
                    ty_le_nhip_lech: float) -> list[dict]:
    """gia_tri: {tên chỉ số: số đo}. Trả về danh sách dòng đã tính trạng thái, độ lệch và đóng góp SHAP."""
    ref = khoang_tham_chieu(thai_nhi, gioi)
    sqi = sqi if np.isfinite(sqi) else 0.5
    dong = []
    for ten, (lo, hi, dv, nguon) in ref.items():
        x = gia_tri.get(ten, np.nan)

        # độ tin cậy: khoảng sóng (đo hình thái) kém chắc chắn hơn nhịp tim;
        # HRV cần ≥ 5 phút và ít nhịp lệch
        if ten in ("PR", "QRS", "QTc (Bazett)"):
            tin_cay = _muc(sqi, 1)
        elif ten in ("SDNN", "RMSSD"):
            tin_cay = _muc(sqi, 0 if thoi_luong_s >= 270 else 1)
            if ty_le_nhip_lech > 0.05:
                tin_cay = "Kém"
        elif ten == "Biến thiên nhịp tim thai":
            tin_cay = _muc(sqi, 0 if thoi_luong_s >= 120 else 1)
        else:
            tin_cay = _muc(sqi)

        if not np.isfinite(x):
            dong.append(dict(ten=ten, x=np.nan, dv=dv, lo=lo, hi=hi, nguon=nguon, trang_thai="Không đánh giá",
                             d=0.0, tin_cay="—", phi=0.0))
            continue
        w = (hi - lo) if (lo is not None and hi is not None) else 0.2 * (hi if hi is not None else lo)
        if lo is not None and x < lo:
            d, tt = (x - lo) / w, "Thấp"
        elif hi is not None and x > hi:
            d, tt = (x - hi) / w, "Cao"
        else:
            d, tt = 0.0, "Bình thường"
        phi = HE_SO_TIN_CAY[tin_cay] * min(abs(d), 2.0)
        dong.append(dict(ten=ten, x=float(x), dv=dv, lo=lo, hi=hi, nguon=nguon, trang_thai=tt,
                         d=float(d), tin_cay=tin_cay, phi=float(phi)))
    return dong


def so(x) -> str:
    """Hiển thị số: 1 chữ số thập phân cho giá trị nhỏ để tránh làm tròn gây hiểu nhầm."""
    return f"{x:.1f}" if abs(x) < 20 else f"{x:.0f}"


def khoang_str(r) -> str:
    if r["lo"] is None:
        return f"≤ {r['hi']:g} {r['dv']}"
    if r["hi"] is None:
        return f"≥ {r['lo']:g} {r['dv']}"
    return f"{r['lo']:g}–{r['hi']:g} {r['dv']}"


def diem_luu_y(dong: list[dict]) -> list[str]:
    """Câu mô tả trung tính (không chẩn đoán) cho các chỉ số nằm ngoài khoảng tham chiếu."""
    out = []
    for r in sorted(dong, key=lambda r: -r["phi"]):
        if r["trang_thai"] in ("Cao", "Thấp"):
            out.append(f"{r['ten']} = {so(r['x'])} {r['dv']}, {'cao hơn' if r['trang_thai'] == 'Cao' else 'thấp hơn'} "
                       f"khoảng tham chiếu {khoang_str(r)} (độ tin cậy phép đo: {r['tin_cay'].lower()}).")
    return out


# ----------------------------------------------------------------------------
# Biểu đồ
# ----------------------------------------------------------------------------
def fig_khoang_tham_chieu(dong: list[dict]):
    """Biểu đồ dạng 'phiếu xét nghiệm': mỗi chỉ số một hàng, dải xanh = khoảng tham chiếu,
    điểm = giá trị đo (đỏ: cao, xanh dương: thấp, xanh lá: bình thường)."""
    n = len(dong)
    fig, ax = plt.subplots(figsize=(11, 0.62 * n + 1.1))
    for i, r in enumerate(dong[::-1]):
        y = i
        if r["lo"] is None:          # chỉ có cận trên: mọi giá trị dưới cận trên đều bình thường
            ax.barh(y, 2, left=-1, height=0.42, color="#c8e6c9", edgecolor="#81c784")
        elif r["hi"] is None:        # chỉ có cận dưới
            ax.barh(y, 2, left=0, height=0.42, color="#c8e6c9", edgecolor="#81c784")
        else:
            ax.barh(y, 1, left=0, height=0.42, color="#c8e6c9", edgecolor="#81c784")
        if not np.isfinite(r["x"]):
            ax.text(0.5, y, "không đo được", ha="center", va="center", fontsize=9, color="#757575")
        else:
            lo = r["lo"] if r["lo"] is not None else r["hi"] - 0.2 * r["hi"] * 2.5
            hi = r["hi"] if r["hi"] is not None else r["lo"] + 0.2 * r["lo"] * 2.5
            pos = (r["x"] - lo) / (hi - lo)
            hien = float(np.clip(pos, -0.95, 1.95))
            ax.plot(hien, y, "o", ms=11, color=MAU[r["trang_thai"]], mec="white", mew=1.5, zorder=3)
            if hien != pos:
                ax.annotate("", xy=(hien + (0.12 if pos > 1 else -0.12), y), xytext=(hien, y),
                            arrowprops=dict(arrowstyle="->", color=MAU[r["trang_thai"]]))
            ax.text(hien, y + 0.33, so(r["x"]), ha="center", fontsize=9, color=MAU[r["trang_thai"]],
                    weight="bold")
        ax.text(2.02, y, khoang_str(r), va="center", fontsize=8.5, color="#424242")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r['ten']} ({r['dv']})" for r in dong[::-1]], fontsize=10)
    ax.set_xlim(-1, 2)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels(["thấp hơn", "cận dưới", "cận trên", "cao hơn"], fontsize=8.5)
    ax.axvline(0, color="#81c784", lw=0.8)
    ax.axvline(1, color="#81c784", lw=0.8)
    ax.set_ylim(-0.6, n - 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Giá trị đo so với khoảng tham chiếu (dải xanh)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def fig_shap(dong: list[dict]):
    """Biểu đồ thác nước kiểu SHAP: từ giá trị nền 0 (mọi chỉ số trong khoảng) cộng dần đóng góp
    của từng chỉ số ra điểm lệch tổng hợp. Đỏ = lệch cao, xanh dương = lệch thấp."""
    ds = sorted(dong, key=lambda r: r["phi"])
    tong = sum(r["phi"] for r in ds)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(ds) + 1.6))
    cong_don = 0.0
    for i, r in enumerate(ds):
        mau = MAU[r["trang_thai"]] if r["phi"] > 0 else "#bdbdbd"
        ax.barh(i, max(r["phi"], 0.004), left=cong_don, color=mau, height=0.6)
        nhan = f"+{r['phi']:.2f}" if r["phi"] > 0 else "0"
        ax.text(cong_don + max(r["phi"], 0) + 0.02, i, nhan, va="center", fontsize=9, color=mau)
        if i < len(ds) - 1 and r["phi"] > 0:
            ax.plot([cong_don + r["phi"]] * 2, [i + 0.3, i + 0.7], color="#757575", lw=0.7)
        cong_don += r["phi"]
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([f"{r['ten']} = {so(r['x'])} {r['dv']}" if np.isfinite(r["x"]) else f"{r['ten']} = —"
                        for r in ds], fontsize=9.5)
    ax.axvline(0, color="#424242", lw=0.8)
    ax.set_xlim(0, max(tong * 1.25, 1))
    ax.set_xlabel("Đóng góp vào điểm lệch (giá trị SHAP)")
    ax.set_title(f"Mức đóng góp của từng chỉ số · giá trị nền E[f] = 0 → điểm lệch tổng hợp f(x) = {tong:.2f}",
                 fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
