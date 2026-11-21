"""Biểu đồ (matplotlib cho báo cáo PDF, Plotly cho dải sóng tương tác) và báo cáo PDF khổ A4."""
import io
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from i18n import t
from so_sanh_chi_so import MAU, khoang_str, so

XANH = "#0b5c7a"        # màu chủ đạo (xanh y tế)
XAM = "#5b6b78"


# ----------------------------------------------------------------------------
# Dải sóng ECG
# ----------------------------------------------------------------------------
def ve_luoi_ecg(ax, t0, t1, ymin, ymax):
    """Lưới giấy điện tim: ô nhỏ 0,04 s × 0,1 mV, ô lớn 0,2 s × 0,5 mV."""
    for x in np.arange(np.floor(t0 / 0.04) * 0.04, t1, 0.04):
        ax.axvline(x, color="#f4c2c2", lw=0.4, zorder=0)
    for y in np.arange(np.floor(ymin / 0.1) * 0.1, ymax, 0.1):
        ax.axhline(y, color="#f4c2c2", lw=0.4, zorder=0)
    for x in np.arange(np.floor(t0 / 0.2) * 0.2, t1, 0.2):
        ax.axvline(x, color="#e08080", lw=0.8, zorder=0)
    for y in np.arange(np.floor(ymin / 0.5) * 0.5, ymax, 0.5):
        ax.axhline(y, color="#e08080", lw=0.8, zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylim(ymin, ymax)


def _doan(ecg, fs, t_start, dur):
    i0, i1 = int(t_start * fs), int(min((t_start + dur) * fs, len(ecg)))
    seg = ecg[i0:i1]
    pad = max(0.15 * (seg.max() - seg.min()), 0.02)
    return i0, i1, np.arange(i0, i1) / fs, seg, seg.min() - pad, seg.max() + pad, pad


def ve_dai_song(ax, ecg_clean, rpeaks, fs, t_start, dur, ref=None):
    i0, i1, tt, seg, ymin, ymax, pad = _doan(ecg_clean, fs, t_start, dur)
    ve_luoi_ecg(ax, tt[0], tt[-1], ymin, ymax)
    ax.plot(tt, seg, color="black", lw=0.9)
    rp = rpeaks[(rpeaks >= i0) & (rpeaks < i1)]
    ax.plot(rp / fs, ecg_clean[rp], "o", color="#1f77b4", ms=3.5)
    if ref is not None:
        rf = ref[(ref >= i0) & (ref < i1)]
        ax.plot(rf / fs, np.full(len(rf), ymax - pad * 0.4), "v", color="#d62728", ms=5)
    return ymin, ymax


def fig_dai_song(ecg_clean, rpeaks, fs, t_start, dur, ref=None):
    _, _, _, _, ymin, ymax, _ = _doan(ecg_clean, fs, t_start, dur)
    w_in, h_in = dur * 25 / 25.4, (ymax - ymin) * 10 / 25.4          # 25 mm/s, 10 mm/mV
    scale = min(1.0, 14 / w_in)
    phong_dai = h_in * scale < 2.2
    fig, ax = plt.subplots(figsize=(w_in * scale, max(h_in * scale, 2.2)))
    ve_dai_song(ax, ecg_clean, rpeaks, fs, t_start, dur, ref)
    tieu_de = (t("Dải sóng ECG") + " (25 mm/s, " + (t("biên độ phóng đại") if phong_dai else "10 mm/mV")
               + ") · " + t("chấm xanh: đỉnh R app tìm được"))
    if ref is not None:
        tieu_de += " · " + t("tam giác đỏ: nhãn chuẩn")
    ax.set_xlabel(t("Thời gian (s)"))
    ax.set_ylabel(t("Biên độ (mV)"))
    ax.set_title(tieu_de)
    fig.tight_layout()
    return fig


def plotly_dai_song(ecg_clean, rpeaks, fs, t0, t1, ref=None, hien_s=10.0):
    """Dải sóng tương tác: lưới giấy điện tim, rê chuột xem giá trị, kéo để dời, cuộn/kéo chọn để phóng to."""
    import plotly.graph_objects as go
    i0, i1 = int(t0 * fs), int(min(t1 * fs, len(ecg_clean)))
    buoc = max(1, int(fs // 500))                 # tối đa ~500 điểm/giây để trình duyệt mượt
    idx = np.arange(i0, i1, buoc)
    tt, y = idx / fs, ecg_clean[idx]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=tt, y=y, mode="lines", line=dict(color="#111", width=1.2), name="ECG",
                               hovertemplate="%{x:.3f} s<br>%{y:.3f} mV<extra></extra>"))
    rp = rpeaks[(rpeaks >= i0) & (rpeaks < i1)]
    if len(rp):
        tr = np.r_[np.nan, 60 * fs / np.diff(rpeaks)]
        hr_rp = tr[np.searchsorted(rpeaks, rp)]
        fig.add_trace(go.Scatter(x=rp / fs, y=ecg_clean[rp], mode="markers", name=t("Đỉnh R"),
                                 marker=dict(color="#1f77b4", size=7, line=dict(color="white", width=1)),
                                 customdata=hr_rp,
                                 hovertemplate=t("Đỉnh R") + " %{x:.3f} s<br>%{customdata:.0f} "
                                 + t("l/p") + "<extra></extra>"))
    rng = float(np.ptp(y)) if len(y) else 1.0
    if ref is not None:
        rf = ref[(ref >= i0) & (ref < i1)]
        fig.add_trace(go.Scatter(x=rf / fs, y=np.full(len(rf), float(np.max(y)) + 0.12 * rng), mode="markers",
                                 name=t("Nhãn chuẩn"), marker=dict(symbol="triangle-down", color="#d62728", size=8),
                                 hovertemplate=t("Nhãn chuẩn") + " %{x:.3f} s<extra></extra>"))
    luoi_chuan = rng >= 0.6                     # biên độ đủ lớn thì dùng lưới 0,5 mV / 0,1 mV
    fig.update_xaxes(title=t("Thời gian (s)"), range=[t0, min(t0 + hien_s, t1)], dtick=0.2, showgrid=True,
                     gridcolor="#e8a0a0", gridwidth=0.8, minor=dict(dtick=0.04, showgrid=True, gridcolor="#f7d6d6"),
                     showticklabels=True, tickformat=".1f", zeroline=False)
    fig.update_yaxes(title=t("Biên độ (mV)"), showgrid=True, gridcolor="#e8a0a0", zeroline=False,
                     **(dict(dtick=0.5, minor=dict(dtick=0.1, showgrid=True, gridcolor="#f7d6d6")) if luoi_chuan else {}))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fff7f7",
                      paper_bgcolor="rgba(0,0,0,0)", dragmode="pan", hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                      font=dict(family="Be Vietnam Pro, sans-serif", size=12))
    # nhãn trục x chỉ hiện mỗi 1 s cho đỡ rối, lưới vẫn giữ 0,2 s
    fig.update_xaxes(tickvals=np.arange(np.floor(t0), t1 + 1, 1.0))
    return fig


def svg_sparkline(hr, band, w=150, h=34):
    """Đường nhịp tim nhỏ (SVG) để đặt trong thẻ chỉ số; vùng xanh nhạt là khoảng bình thường."""
    hr = np.asarray(hr, float)
    hr = hr[np.isfinite(hr)]
    if len(hr) < 2:
        return ""
    if len(hr) > 120:
        hr = np.array([c.mean() for c in np.array_split(hr, 120)])
    lo, hi = min(hr.min(), band[0]) - 5, max(hr.max(), band[1]) + 5
    y = lambda v: h - (v - lo) / (hi - lo) * h
    x = np.linspace(0, w, len(hr))
    pts = " ".join(f"{a:.1f},{y(b):.1f}" for a, b in zip(x, hr))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'<rect x="0" y="{y(band[1]):.1f}" width="{w}" height="{y(band[0]) - y(band[1]):.1f}" fill="#2e7d32" '
            f'opacity="0.10"/><polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'vector-effect="non-scaling-stroke"/></svg>')


# ----------------------------------------------------------------------------
# Biểu đồ khác
# ----------------------------------------------------------------------------
def fig_truoc_sau_loc(raw, clean, fs, dur=5):
    n = int(min(dur * fs, len(raw)))
    tt = np.arange(n) / fs
    fig, axs = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    axs[0].plot(tt, raw[:n], color="gray", lw=0.8)
    axs[0].set_title(t("Tín hiệu gốc"))
    axs[1].plot(tt, clean[:n], color="crimson", lw=0.9)
    axs[1].set_title(t("Sau khi lọc"))
    axs[1].set_xlabel(t("Thời gian (s)"))
    for a in axs:
        a.set_ylabel("mV")
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_pho(raw, clean, fs):
    from scipy.signal import welch
    n = int(min(len(raw), 4 * fs))
    f1, p1 = welch(raw - np.mean(raw), fs=fs, nperseg=n)
    f2, p2 = welch(clean - np.mean(clean), fs=fs, nperseg=n)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.semilogy(f1, p1 + 1e-20, color="gray", lw=0.9, label=t("Trước lọc"))
    ax.semilogy(f2, p2 + 1e-20, color="crimson", lw=1.1, label=t("Sau lọc"))
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel(t("Tần số (Hz)"))
    ax.set_ylabel(t("Mật độ công suất (mV²/Hz)"))
    ax.set_title(t("Phổ công suất (Welch) trước và sau lọc"))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_dap_ung(fs, cfg):
    """Đáp ứng biên độ của phần lọc tuyến tính (Butterworth 2 chiều → |H|², nhân Notch)."""
    from scipy import signal as ss
    nyq = fs / 2
    hi = min(cfg["hi"], 0.99 * nyq)
    f = np.linspace(0.01, nyq, 4000)
    if cfg["nen"].startswith("Wavelet"):
        sos = ss.butter(cfg["bac"], hi, btype="lowpass", fs=fs, output="sos")
    else:
        sos = ss.butter(cfg["bac"], [cfg["lo"], hi], btype="bandpass", fs=fs, output="sos")
    _, h = ss.sosfreqz(sos, worN=f, fs=fs)
    H = np.abs(h) ** 2
    if cfg["notch"]:
        for k in ([1, 2, 3] if cfg["hai"] else [1]):
            f0 = cfg["f_notch"] * k
            if f0 < 0.99 * nyq:
                b, a = ss.iirnotch(f0, cfg["Q"], fs=fs)
                _, hn = ss.freqz(b, a, worN=f, fs=fs)
                H = H * np.abs(hn) ** 2
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(f, 20 * np.log10(H + 1e-12), color="#1f77b4")
    ax.axhline(-3, color="gray", ls="--", lw=0.8)
    ax.text(nyq * 0.99, -3, "−3 dB", ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, nyq)
    ax.set_xlabel(t("Tần số (Hz)"))
    ax.set_ylabel(t("Biên độ (dB)"))
    ax.set_title(t("Đáp ứng tần số của bộ lọc (Butterworth lọc 2 chiều + Notch)"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_nhip_tim(rpeaks, hr, fs, band=(60, 100)):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rpeaks[1:] / fs, hr, color="crimson", lw=1.2)
    ax.axhspan(band[0], band[1], color="green", alpha=0.08, label=t("Ngưỡng {a}–{b} l/p", a=band[0], b=band[1]))
    ax.set_xlabel(t("Thời gian (s)"))
    ax.set_ylabel(t("Nhịp tim (l/p)"))
    ax.set_title(t("Nhịp tim tức thời theo thời gian"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_rr(rr_ms):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(rr_ms, bins=30, color="#1f77b4", edgecolor="white")
    axs[0].set_xlabel(t("Khoảng RR (ms)"))
    axs[0].set_ylabel(t("Số nhịp"))
    axs[0].set_title(t("Phân bố khoảng RR"))
    axs[1].scatter(rr_ms[:-1], rr_ms[1:], s=10, alpha=0.6, color="#ff7f0e")
    lo, hi = rr_ms.min() * 0.95, rr_ms.max() * 1.05
    axs[1].plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    axs[1].set_xlabel("RRₙ (ms)")
    axs[1].set_ylabel("RRₙ₊₁ (ms)")
    axs[1].set_title(t("Biểu đồ Poincaré"))
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Báo cáo PDF khổ A4 kiểu phiếu kết quả bệnh viện
# ----------------------------------------------------------------------------
def _logo(fig, x, y, s):
    """Biểu tượng: ô vuông bo góc màu chủ đạo với một nhịp ECG trắng."""
    ax = fig.add_axes([x, y, s * 11.69 / 8.27 * 0.707, s])
    ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=XANH, ec="none"))
    px = [0.12, 0.34, 0.42, 0.5, 0.58, 0.66, 0.88]
    py = [0.5, 0.5, 0.38, 0.84, 0.22, 0.5, 0.5]
    ax.plot(px, py, color="white", lw=2.2, solid_joinstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def tao_pdf(tt: dict, kpi: list, dong_ss: list, luu_y: list, strip: dict, figs: list,
            don_vi_ten="LAB TIN HỌC VẬT LÝ KỸ THUẬT", don_vi_phu="Trường Đại học Bách Khoa – ĐHQG TP.HCM"):
    """Trang 1: phiếu kết quả (đầu phiếu, hành chính, ô chỉ số, dải sóng 10 s, bảng so sánh, lưu ý, ô ký).
    Các trang sau: biểu đồ chi tiết.
    tt: thông tin hành chính {nhãn: giá trị}; kpi: [(nhãn, giá trị, đơn vị, trạng thái)];
    strip: dict(ecg, rpeaks, fs, t0, ref)."""
    buf = io.BytesIO()
    W = 8.27
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(W, 11.69))
        # --- Đầu phiếu
        fig.add_artist(Rectangle((0, 0.935), 1, 0.065, transform=fig.transFigure, fc=XANH, ec="none"))
        _logo(fig, 0.05, 0.943, 0.048)
        fig.text(0.125, 0.973, t(don_vi_ten), color="white", fontsize=11, weight="bold", va="center")
        fig.text(0.125, 0.952, t(don_vi_phu), color="#d6eef6", fontsize=8.5, va="center")
        fig.text(0.95, 0.973, t("Mã phiếu") + f": {tt.get('_ma', '')}", color="white", fontsize=8.5,
                 ha="right", va="center")
        fig.text(0.95, 0.952, tt.get("_ngay", ""), color="#d6eef6", fontsize=8.5, ha="right", va="center")
        fig.text(0.5, 0.905, t("PHIẾU KẾT QUẢ PHÂN TÍCH ĐIỆN TÂM ĐỒ"), ha="center", fontsize=15, weight="bold",
                 color="#1b2733")

        # --- Thông tin hành chính (2 cột)
        y0 = 0.885
        fig.add_artist(Rectangle((0.05, y0 - 0.088), 0.90, 0.088, transform=fig.transFigure, fc="#f4f7f9",
                                 ec="#d5dde3", lw=0.8))
        muc = [(k, v) for k, v in tt.items() if not k.startswith("_")]
        nua = (len(muc) + 1) // 2
        for i, (k, v) in enumerate(muc):
            cx = 0.065 if i < nua else 0.51
            cy = y0 - 0.017 - (i % nua) * 0.0165
            fig.text(cx, cy, f"{k}", fontsize=8, color=XAM, va="center")
            fig.text(cx + 0.145, cy, textwrap.shorten(str(v), 42, placeholder="…"), fontsize=8.6,
                     color="#1b2733", va="center", weight="bold" if i == 0 else "normal")

        # --- Ô chỉ số chính
        yk, hk = 0.715, 0.068
        n = len(kpi)
        wk = (0.90 - 0.012 * (n - 1)) / n
        for i, (nhan, gt, dv, tt_) in enumerate(kpi):
            x = 0.05 + i * (wk + 0.012)
            mau = MAU.get(tt_, "#9e9e9e") if tt_ else XANH
            fig.add_artist(FancyBboxPatch((x, yk), wk, hk, transform=fig.transFigure, fc="white", ec="#d5dde3",
                                          lw=0.8, boxstyle="round,pad=0,rounding_size=0.006"))
            fig.add_artist(Rectangle((x, yk), 0.005, hk, transform=fig.transFigure, fc=mau, ec="none"))
            fig.text(x + 0.014, yk + hk - 0.016, nhan, fontsize=7.6, color=XAM)
            fig.text(x + 0.014, yk + 0.02, gt, fontsize=15, weight="bold", color="#1b2733")
            fig.text(x + wk - 0.01, yk + 0.022, dv, fontsize=7.6, color=XAM, ha="right")
            if tt_:
                fig.text(x + wk - 0.01, yk + hk - 0.016, t(tt_), fontsize=7.4, color=mau, ha="right",
                         weight="bold")

        # --- Dải sóng 10 s
        fig.text(0.05, 0.69, t("DẢI SÓNG ĐIỆN TIM 10 GIÂY"), fontsize=9.5, weight="bold", color=XANH)
        fig.text(0.95, 0.69, t("25 mm/s · lưới 0,2 s × 0,5 mV · chấm xanh: đỉnh R"), fontsize=7.4, color=XAM,
                 ha="right")
        ax = fig.add_axes([0.05, 0.555, 0.90, 0.128])
        dur = min(10.0, len(strip["ecg"]) / strip["fs"] - strip["t0"])
        ve_dai_song(ax, strip["ecg"], strip["rpeaks"], strip["fs"], strip["t0"], dur, strip.get("ref"))
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e08080")

        # --- Bảng so sánh với khoảng tham chiếu
        y = 0.525
        fig.text(0.05, y, t("SO SÁNH VỚI KHOẢNG THAM CHIẾU"), fontsize=9.5, weight="bold", color=XANH)
        y -= 0.022
        cot = [(0.05, t("Chỉ số")), (0.30, t("Kết quả")), (0.45, t("Khoảng tham chiếu")),
               (0.63, t("Đánh giá")), (0.76, t("Độ tin cậy đo"))]
        fig.add_artist(Rectangle((0.05, y - 0.006), 0.90, 0.02, transform=fig.transFigure, fc="#e7eef2", ec="none"))
        for xc, c in cot:
            fig.text(xc + 0.008, y + 0.004, c, fontsize=8.2, weight="bold", color="#1b2733", va="center")
        y -= 0.021
        for j, r in enumerate(dong_ss):
            if j % 2:
                fig.add_artist(Rectangle((0.05, y - 0.006), 0.90, 0.019, transform=fig.transFigure, fc="#f7f9fa",
                                         ec="none"))
            gt = "—" if not np.isfinite(r["x"]) else f"{so(r['x'])} {t(r['dv'])}"
            mau = MAU.get(r["trang_thai"], "black") if r["trang_thai"] in ("Cao", "Thấp") else "#1b2733"
            o = [t(r["ten"]), gt, khoang_str(r), t(r["trang_thai"]), t(r["tin_cay"])]
            for (xc, _), v in zip(cot, o):
                dam = xc in (0.30, 0.63) and r["trang_thai"] in ("Cao", "Thấp")
                fig.text(xc + 0.008, y + 0.003, v, fontsize=8.2, va="center",
                         color=mau if xc in (0.30, 0.63) else "#1b2733", weight="bold" if dam else "normal")
            y -= 0.019
        fig.add_artist(plt.Line2D([0.05, 0.95], [y + 0.01, y + 0.01], transform=fig.transFigure, color="#d5dde3",
                                  lw=0.8))

        # --- Điểm cần lưu ý
        y -= 0.022
        fig.text(0.05, y, t("CÁC ĐIỂM CẦN BÁC SĨ LƯU Ý"), fontsize=9.5, weight="bold", color=XANH)
        y -= 0.02
        for line in (luu_y or [t("Tất cả chỉ số đánh giá được đều nằm trong khoảng tham chiếu.")])[:6]:
            for i, d in enumerate(textwrap.wrap(line, 108)):
                fig.text(0.06 if i == 0 else 0.072, y, ("• " if i == 0 else "") + d, fontsize=8.3, color="#1b2733")
                y -= 0.0165
            y -= 0.003

        # --- Ô kết luận và chữ ký
        yb = 0.075
        fig.add_artist(Rectangle((0.05, yb), 0.56, 0.135, transform=fig.transFigure, fc="white", ec="#9fb0bc",
                                 lw=0.8))
        fig.text(0.062, yb + 0.118, t("KẾT LUẬN CỦA BÁC SĨ"), fontsize=8.6, weight="bold", color=XANH)
        for k in range(4):
            fig.add_artist(plt.Line2D([0.062, 0.598], [yb + 0.09 - k * 0.024] * 2, transform=fig.transFigure,
                                      color="#c7d1d8", lw=0.6, ls=(0, (1, 2))))
        fig.add_artist(Rectangle((0.63, yb), 0.32, 0.135, transform=fig.transFigure, fc="white", ec="#9fb0bc",
                                 lw=0.8))
        fig.text(0.79, yb + 0.118, t("Ngày ..... tháng ..... năm 20....."), fontsize=8, ha="center", color=XAM)
        fig.text(0.79, yb + 0.098, t("BÁC SĨ ĐỌC KẾT QUẢ"), fontsize=8.6, weight="bold", ha="center", color="#1b2733")
        fig.text(0.79, yb + 0.083, t("(Ký, ghi rõ họ tên)"), fontsize=7.6, ha="center", style="italic", color=XAM)

        fig.add_artist(plt.Line2D([0.05, 0.95], [0.055, 0.055], transform=fig.transFigure, color="#d5dde3", lw=0.6))
        fig.text(0.05, 0.04, t("Kết quả tự động chỉ dùng cho học tập/nghiên cứu, không thay thế chẩn đoán y khoa."),
                 fontsize=7.2, style="italic", color=XAM)
        fig.text(0.95, 0.04, t("Trang") + f" 1/{1 + len(figs)}", fontsize=7.2, color=XAM, ha="right")
        pdf.savefig(fig)
        plt.close(fig)

        for k, f in enumerate(figs, 2):
            f.text(0.99, 0.005, t("Trang") + f" {k}/{1 + len(figs)}", fontsize=7, color=XAM, ha="right")
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf
