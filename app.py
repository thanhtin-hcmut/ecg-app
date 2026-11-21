"""
App phân tích tín hiệu ECG - Lab Tin học Vật lý Kỹ thuật
Chạy:  streamlit run app.py
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import neurokit2 as nk
import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_physionet import doc_edf, doc_nhan_wfdb, doc_wfdb

# ----------------------------------------------------------------------------
# Cấu hình trang
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Phân tích ECG", page_icon="🫀", layout="wide")
st.title("🫀 Ứng dụng phân tích tín hiệu ECG")
st.caption("Lab Tin học Vật lý Kỹ thuật · Kết quả chỉ dùng cho học tập/nghiên cứu, không dùng để chẩn đoán.")


# ----------------------------------------------------------------------------
# 1. HÀM XỬ LÝ
# ----------------------------------------------------------------------------
def _la_so(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def doc_file(file) -> pd.DataFrame:
    """Đọc CSV/TXT: tự nhận dấu phân cách và có/không có dòng tiêu đề.
    Hỗ trợ cả file chỉ có 1 cột số (vd. dữ liệu ADC từ Arduino/ESP32)."""
    text = file.getvalue().decode("utf-8-sig", errors="ignore")
    dong = [l for l in text.splitlines() if l.strip()][:20]
    if not dong:
        raise ValueError("File rỗng.")
    mau = dong[1] if len(dong) > 1 else dong[0]
    sep = next((d for d in [";", "\t", ","] if d in mau), r"\s+")
    tach = lambda l: [x for x in (l.split(sep) if sep != r"\s+" else l.split()) if x.strip() != ""]
    co_tieu_de = not all(_la_so(x.strip()) for x in tach(dong[0]))
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                     header=0 if co_tieu_de else None)
    if not co_tieu_de:
        df.columns = ["ecg"] if df.shape[1] == 1 else [f"kenh_{i+1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df.dropna(axis=1, how="all")


DUOI_NHAN = ("atr", "qrs", "ann")


def doc_nhieu_file(files) -> tuple[pd.DataFrame, dict]:
    """Đọc 1 hoặc nhiều file: CSV/TXT, EDF/EDF+ (.edf), WFDB (.hea + .dat) và nhãn WFDB (.atr/.qrs/.ann).
    Trả về (bảng tín hiệu, thông tin: fs, đơn vị từng kênh, vị trí đỉnh R chuẩn nếu có, mô tả định dạng)."""
    duoi = lambda f: f.name.lower().rsplit(".", 1)[-1]
    edf = [f for f in files if duoi(f) == "edf"]
    hea = [f for f in files if duoi(f) == "hea"]
    nhan = [f for f in files if duoi(f) in DUOI_NHAN]
    bang = [f for f in files if duoi(f) in ("csv", "txt")]
    meta = dict(fs=None, units={}, ref=None, ky_hieu=None, mo_ta="", nguon_nhan="")

    if edf:
        d = doc_edf(edf[0].getvalue())
        meta["mo_ta"] = f"EDF/EDF+ ({edf[0].name})"
    elif hea:
        dat = {f.name: f.getvalue() for f in files if duoi(f) == "dat"}
        d = doc_wfdb(hea[0].getvalue().decode("latin-1"), dat)
        meta["mo_ta"] = f"WFDB ({hea[0].name})"
    elif bang:
        return doc_file(bang[0]), meta
    elif any(duoi(f) == "dat" for f in files):
        raise ValueError("File .dat của PhysioNet cần tải lên kèm file .hea cùng tên.")
    else:
        raise ValueError("Chưa có file tín hiệu. Hãy tải .csv/.txt, .edf, hoặc cặp .hea + .dat.")

    if not d["signals"]:
        raise ValueError("File không có kênh tín hiệu nào.")
    # chỉ giữ các kênh có cùng tần số lấy mẫu phổ biến nhất
    fs_ds = list(d["fs"].values())
    fs = max(set(fs_ds), key=fs_ds.count)
    ten = [k for k in d["signals"] if d["fs"][k] == fs]
    n = min(len(d["signals"][k]) for k in ten)
    df = pd.DataFrame({k: d["signals"][k][:n] for k in ten})
    meta.update(fs=fs, units={k: d["units"][k] for k in ten})

    if nhan:
        mau, ky = doc_nhan_wfdb(nhan[0].getvalue())
        meta.update(ref=mau[mau < n], ky_hieu=ky, nguon_nhan=nhan[0].name)
    elif d["ann_s"]:
        mau = np.round(np.asarray(d["ann_s"]) * fs).astype(int)
        meta.update(ref=mau[(mau >= 0) & (mau < n)], nguon_nhan="nhãn EDF+ trong file")
    return df, meta


def doi_chieu(det, ref, fs, dung_sai_ms):
    """So khớp đỉnh R tìm được với nhãn chuẩn (mỗi nhãn khớp tối đa 1 đỉnh, trong ± dung sai)."""
    tol = dung_sai_ms * fs / 1000
    det = np.sort(np.asarray(det))
    da_dung = np.zeros(len(det), bool)
    tp, lech = 0, []
    for r in np.sort(np.asarray(ref)):
        j = np.searchsorted(det, r)
        ung_vien = [k for k in (j - 1, j) if 0 <= k < len(det) and not da_dung[k] and abs(det[k] - r) <= tol]
        if ung_vien:
            k = min(ung_vien, key=lambda k: abs(det[k] - r))
            da_dung[k] = True
            tp += 1
            lech.append(abs(det[k] - r) / fs * 1000)
    fn, fp = len(ref) - tp, len(det) - tp
    se = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * se * ppv / (se + ppv) if se + ppv else np.nan
    return dict(TP=tp, FN=fn, FP=fp, Se=se, PPV=ppv, F1=f1, lech_ms=float(np.mean(lech)) if lech else np.nan)


PP_TUY_CHINH = "Tùy chỉnh (Butterworth + Notch + Wavelet)"


def _nguong_wavelet(d, sigma, kieu):
    """Ngưỡng cho một mức chi tiết: Universal (VisuShrink) hoặc BayesShrink."""
    if kieu.startswith("Universal"):
        return sigma * np.sqrt(2 * np.log(len(d)))
    sigma_x = np.sqrt(max(np.var(d) - sigma ** 2, 1e-12))
    return sigma ** 2 / sigma_x


def loc_tuy_chinh(x: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Chuỗi lọc tùy chỉnh (tất cả đều không lệch pha):
    1) Khử trôi đường nền: Butterworth thông cao hoặc Wavelet (bỏ thành phần xấp xỉ tần thấp)
    2) Butterworth thông thấp/thông dải với bậc và tần số cắt do người dùng chọn
    3) Notch 50/60 Hz (tùy chọn kèm hài bậc 2, 3)
    4) Khử nhiễu bằng wavelet (DWT, ngưỡng mềm/cứng theo từng mức)"""
    import pywt
    from scipy import signal as ss

    y = np.asarray(x, dtype=float) - np.median(x)
    nyq = fs / 2
    lo, hi = cfg["lo"], min(cfg["hi"], 0.99 * nyq)

    # 1) Khử trôi nền
    if cfg["nen"].startswith("Wavelet"):
        w = pywt.Wavelet(cfg["wavelet"])
        # chọn mức sao cho dải xấp xỉ [0, fs/2^(L+1)] nằm dưới tần số cắt dưới
        L = int(np.ceil(np.log2(fs / lo))) - 1
        L = int(np.clip(L, 1, pywt.dwt_max_level(len(y), w.dec_len)))
        c = pywt.wavedec(y, w, level=L)
        c[0] = np.zeros_like(c[0])
        y = pywt.waverec(c, w)[: len(x)]
        sos = ss.butter(cfg["bac"], hi, btype="lowpass", fs=fs, output="sos")
    else:
        sos = ss.butter(cfg["bac"], [lo, hi], btype="bandpass", fs=fs, output="sos")

    # 2) Butterworth (sosfiltfilt: lọc 2 chiều → không lệch pha, bậc hiệu dụng gấp đôi)
    y = ss.sosfiltfilt(sos, y)

    # 3) Notch điện lưới
    if cfg["notch"]:
        hai = [1, 2, 3] if cfg["hai"] else [1]
        for k in hai:
            f0 = cfg["f_notch"] * k
            if f0 < 0.99 * nyq:
                b, a = ss.iirnotch(f0, cfg["Q"], fs=fs)
                y = ss.filtfilt(b, a, y)

    # 4) Khử nhiễu wavelet
    if cfg["wl_on"]:
        w = pywt.Wavelet(cfg["wavelet"])
        L = min(cfg["muc"], pywt.dwt_max_level(len(y), w.dec_len))
        c = pywt.wavedec(y, w, level=L)
        sigma1 = np.median(np.abs(c[-1])) / 0.6745          # ước lượng nhiễu từ mức chi tiết mịn nhất
        for i in range(1, len(c)):
            sigma = np.median(np.abs(c[i])) / 0.6745 if cfg["theo_muc"] else sigma1
            t = _nguong_wavelet(c[i], sigma, cfg["nguong"])
            c[i] = pywt.threshold(c[i], t, mode="soft" if cfg["mem"] else "hard")
        y = pywt.waverec(c, w)[: len(x)]
    return y


def mo_ta_loc(method: str, cfg: dict) -> str:
    if method != PP_TUY_CHINH:
        return method
    s = []
    if cfg["nen"].startswith("Wavelet"):
        s.append(f"khử nền wavelet (<{cfg['lo']:g} Hz) + Butterworth thông thấp bậc {cfg['bac']}, {cfg['hi']:g} Hz")
    else:
        s.append(f"Butterworth thông dải bậc {cfg['bac']}, {cfg['lo']:g}–{cfg['hi']:g} Hz")
    if cfg["notch"]:
        s.append(f"Notch {cfg['f_notch']} Hz (Q={cfg['Q']:g}{', kèm hài' if cfg['hai'] else ''})")
    if cfg["wl_on"]:
        s.append(f"wavelet {cfg['wavelet']} mức {cfg['muc']}, ngưỡng {cfg['nguong'].split(' ')[0]} "
                 f"{'mềm' if cfg['mem'] else 'cứng'}")
    return "; ".join(s)


def xu_ly_ecg(ecg_raw: np.ndarray, fs: int, method: str, cfg: dict | None = None):
    """Lọc tín hiệu, tìm đỉnh R, tính nhịp tim và chất lượng tín hiệu."""
    if method == PP_TUY_CHINH:
        ecg_clean = loc_tuy_chinh(ecg_raw, fs, cfg)
    else:
        ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=fs, method=method)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, correct_artifacts=True)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(rpeaks) < 3:
        raise ValueError("Tìm được quá ít đỉnh R. Kiểm tra lại cột dữ liệu và tần số lấy mẫu.")
    rr_ms = np.diff(rpeaks) / fs * 1000
    hr = 60000 / rr_ms
    try:
        quality = nk.ecg_quality(ecg_clean, rpeaks=rpeaks, sampling_rate=fs)
        quality = float(np.nanmean(quality))
    except Exception:
        quality = np.nan
    return ecg_clean, rpeaks, rr_ms, hr, quality


def tinh_khoang_song(ecg_clean, rpeaks, fs):
    """Ước tính PR, QRS, QT, QTc (Bazett) bằng phân tích hình thái sóng.
    Chỉ dùng tối đa 300 nhịp đầu (đủ để lấy trung vị) cho nhanh với bản ghi dài."""
    if len(rpeaks) > 300:
        rpeaks = rpeaks[:300]
        ecg_clean = ecg_clean[: rpeaks[-1] + int(fs)]
    try:
        _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method="dwt")
    except Exception:
        return {}

    def trung_vi(a, b):
        a = np.asarray(waves.get(a, []), dtype=float)
        b = np.asarray(waves.get(b, []), dtype=float)
        n = min(len(a), len(b))
        if n == 0:
            return np.nan
        d = (b[:n] - a[:n]) / fs * 1000
        d = d[(d > 0) & np.isfinite(d)]
        return float(np.median(d)) if len(d) else np.nan

    # Dùng đỉnh Q/S (ổn định hơn mốc onset/offset của DWT, vốn hay làm QRS rộng giả)
    pr = trung_vi("ECG_P_Onsets", "ECG_R_Onsets")
    qrs = trung_vi("ECG_Q_Peaks", "ECG_S_Peaks")
    qt = trung_vi("ECG_Q_Peaks", "ECG_T_Offsets")
    rr_s = np.median(np.diff(rpeaks)) / fs
    qtc_b = qt / np.sqrt(rr_s) if np.isfinite(qt) else np.nan          # Bazett
    qtc_f = qt / np.cbrt(rr_s) if np.isfinite(qt) else np.nan          # Fridericia
    return {"PR (ms)": pr, "QRS (ms)": qrs, "QT (ms)": qt,
            "QTc Bazett (ms)": qtc_b, "QTc Fridericia (ms)": qtc_f}


def tinh_hrv(rpeaks, fs):
    """Chỉ số HRV miền thời gian và Poincaré."""
    hrv = nk.hrv_time(rpeaks, sampling_rate=fs)
    lay = lambda df, c: float(df[c].iloc[0]) if c in df else np.nan
    # Poincaré tính trực tiếp: SD1 = sqrt(0,5)·SD(ΔRR), SD2 = sqrt(2·SDNN² − SD1²)
    rr = np.diff(rpeaks) / fs * 1000
    sd1 = np.sqrt(0.5) * np.std(np.diff(rr), ddof=1)
    sd2 = np.sqrt(max(2 * np.var(rr, ddof=1) - sd1 ** 2, 0))
    return {
        "MeanNN (ms)": lay(hrv, "HRV_MeanNN"),
        "SDNN (ms)": lay(hrv, "HRV_SDNN"),
        "RMSSD (ms)": lay(hrv, "HRV_RMSSD"),
        "pNN50 (%)": lay(hrv, "HRV_pNN50"),
        "SD1 (ms)": float(sd1),
        "SD2 (ms)": float(sd2),
    }


def nguong_nhip(thai_nhi: bool):
    """Ngưỡng nhịp tim bình thường: người lớn khi nghỉ 60–100 l/p; thai nhi (FIGO) 110–160 l/p."""
    return (110, 160) if thai_nhi else (60, 100)


def danh_gia(hr_mean, rr_ms, khoang, thai_nhi=False):
    """Nhận xét tự động dựa trên ngưỡng tham khảo (người lớn khi nghỉ, hoặc thai nhi theo FIGO)."""
    nx = []
    lo, hi = nguong_nhip(thai_nhi)
    doi_tuong = "nhịp tim thai" if thai_nhi else "nhịp tim"
    if hr_mean < lo:
        nx.append(f"{doi_tuong.capitalize()} trung bình {hr_mean:.0f} l/p: chậm hơn ngưỡng {lo}–{hi} l/p.")
    elif hr_mean > hi:
        nx.append(f"{doi_tuong.capitalize()} trung bình {hr_mean:.0f} l/p: nhanh hơn ngưỡng {lo}–{hi} l/p.")
    else:
        nx.append(f"{doi_tuong.capitalize()} trung bình {hr_mean:.0f} l/p: trong ngưỡng {lo}–{hi} l/p.")

    cv = np.std(rr_ms) / np.mean(rr_ms) * 100
    if cv > 15:
        nx.append(f"Khoảng RR dao động mạnh (CV = {cv:.1f}%): nhịp không đều, nên xem lại sóng.")
    else:
        nx.append(f"Khoảng RR tương đối đều (CV = {cv:.1f}%).")

    lech = np.abs(np.diff(rr_ms)) / rr_ms[:-1]
    n_bat_thuong = int(np.sum(lech > 0.2))
    if n_bat_thuong:
        nx.append(f"Có {n_bat_thuong} nhịp có RR lệch >20% so với nhịp trước (nghi ngoại tâm thu hoặc nhiễu).")

    if thai_nhi:
        nx.append("Các khoảng PR/QRS/QT của thai nhi không được đánh giá theo ngưỡng người lớn.")
        return nx
    nguong = {"PR (ms)": (120, 200), "QRS (ms)": (60, 120), "QTc Bazett (ms)": (350, 450)}
    for k, (lo, hi) in nguong.items():
        v = khoang.get(k, np.nan)
        if np.isfinite(v):
            tt = "trong ngưỡng" if lo <= v <= hi else "ngoài ngưỡng"
            nx.append(f"{k.split(' (')[0]} = {v:.0f} ms: {tt} tham khảo {lo}–{hi} ms.")
    return nx


# ----------------------------------------------------------------------------
# 2. HÀM VẼ
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


def fig_dai_song(ecg_clean, rpeaks, fs, t_start, dur, ref=None):
    i0, i1 = int(t_start * fs), int(min((t_start + dur) * fs, len(ecg_clean)))
    t = np.arange(i0, i1) / fs
    seg = ecg_clean[i0:i1]
    pad = max(0.15 * (seg.max() - seg.min()), 0.02)
    ymin, ymax = seg.min() - pad, seg.max() + pad
    # 25 mm/s, 10 mm/mV  -> 1 s = 25 mm, 1 mV = 10 mm (tỷ lệ khung hình đúng chuẩn)
    w_in = dur * 25 / 25.4
    h_in = (ymax - ymin) * 10 / 25.4
    scale = min(1.0, 14 / w_in)
    phong_dai = h_in * scale < 2.2           # tín hiệu biên độ nhỏ (vd. ECG thai nhi) -> phóng to trục dọc
    fig, ax = plt.subplots(figsize=(w_in * scale, max(h_in * scale, 2.2)))
    ve_luoi_ecg(ax, t[0], t[-1], ymin, ymax)
    ax.plot(t, seg, color="black", lw=1)
    rp = rpeaks[(rpeaks >= i0) & (rpeaks < i1)]
    ax.plot(rp / fs, ecg_clean[rp], "o", color="#1f77b4", ms=4)
    tieu_de = ("Dải sóng ECG (25 mm/s, " + ("biên độ phóng đại" if phong_dai else "10 mm/mV")
               + ") · chấm xanh: đỉnh R app tìm được")
    if ref is not None:
        rf = ref[(ref >= i0) & (ref < i1)]
        ax.plot(rf / fs, np.full(len(rf), ymax - pad * 0.4), "v", color="#d62728", ms=6)
        tieu_de += " · tam giác đỏ: nhãn chuẩn"
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Biên độ (mV)")
    ax.set_title(tieu_de)
    fig.tight_layout()
    return fig


def fig_truoc_sau_loc(raw, clean, fs, dur=5):
    n = int(min(dur * fs, len(raw)))
    t = np.arange(n) / fs
    fig, axs = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    axs[0].plot(t, raw[:n], color="gray", lw=0.8)
    axs[0].set_title("Tín hiệu gốc")
    axs[1].plot(t, clean[:n], color="crimson", lw=0.9)
    axs[1].set_title("Sau khi lọc")
    axs[1].set_xlabel("Thời gian (s)")
    for a in axs:
        a.set_ylabel("mV")
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_pho(raw, clean, fs):
    """Phổ công suất (Welch) trước và sau lọc."""
    from scipy.signal import welch
    n = int(min(len(raw), 4 * fs))
    f1, p1 = welch(raw - np.mean(raw), fs=fs, nperseg=n)
    f2, p2 = welch(clean - np.mean(clean), fs=fs, nperseg=n)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.semilogy(f1, p1 + 1e-20, color="gray", lw=0.9, label="Trước lọc")
    ax.semilogy(f2, p2 + 1e-20, color="crimson", lw=1.1, label="Sau lọc")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Tần số (Hz)")
    ax.set_ylabel("Mật độ công suất (mV²/Hz)")
    ax.set_title("Phổ công suất (Welch) trước và sau lọc")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_dap_ung(fs, cfg):
    """Đáp ứng biên độ của phần lọc tuyến tính (Butterworth 2 chiều + Notch)."""
    from scipy import signal as ss
    nyq = fs / 2
    hi = min(cfg["hi"], 0.99 * nyq)
    f = np.linspace(0.01, nyq, 4000)
    if cfg["nen"].startswith("Wavelet"):
        sos = ss.butter(cfg["bac"], hi, btype="lowpass", fs=fs, output="sos")
    else:
        sos = ss.butter(cfg["bac"], [cfg["lo"], hi], btype="bandpass", fs=fs, output="sos")
    _, h = ss.sosfreqz(sos, worN=f, fs=fs)
    H = np.abs(h) ** 2                      # lọc 2 chiều → |H|²
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
    ax.set_xlabel("Tần số (Hz)")
    ax.set_ylabel("Biên độ (dB)")
    ax.set_title("Đáp ứng tần số của bộ lọc (Butterworth lọc 2 chiều + Notch)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_nhip_tim(rpeaks, hr, fs, band=(60, 100)):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rpeaks[1:] / fs, hr, color="crimson", lw=1.2)
    ax.axhspan(band[0], band[1], color="green", alpha=0.08, label=f"Ngưỡng {band[0]}–{band[1]} l/p")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Nhịp tim (l/p)")
    ax.set_title("Nhịp tim tức thời theo thời gian")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_rr(rr_ms):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(rr_ms, bins=30, color="#1f77b4", edgecolor="white")
    axs[0].set_xlabel("Khoảng RR (ms)")
    axs[0].set_ylabel("Số nhịp")
    axs[0].set_title("Phân bố khoảng RR")
    axs[1].scatter(rr_ms[:-1], rr_ms[1:], s=10, alpha=0.6, color="#ff7f0e")
    lo, hi = rr_ms.min() * 0.95, rr_ms.max() * 1.05
    axs[1].plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    axs[1].set_xlabel("RRₙ (ms)")
    axs[1].set_ylabel("RRₙ₊₁ (ms)")
    axs[1].set_title("Biểu đồ Poincaré")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def tao_pdf(thong_tin, bang_chi_so, nhan_xet, figs):
    """Báo cáo PDF: trang 1 là thông tin + chỉ số + nhận xét, các trang sau là biểu đồ."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        y = 0.95
        fig.text(0.5, y, "BÁO CÁO PHÂN TÍCH ECG", ha="center", fontsize=16, weight="bold")
        y -= 0.05
        import textwrap
        for k, v in thong_tin.items():
            for i, dong in enumerate(textwrap.wrap(f"{k}: {v}", 95)):
                fig.text(0.08 if i == 0 else 0.10, y, dong, fontsize=10)
                y -= 0.022
        y -= 0.02
        fig.text(0.08, y, "CHỈ SỐ ĐO", fontsize=12, weight="bold")
        y -= 0.03
        for _, r in bang_chi_so.iterrows():
            fig.text(0.10, y, f"{r['Chỉ số']}", fontsize=10)
            fig.text(0.55, y, f"{r['Giá trị']}", fontsize=10)
            y -= 0.022
        y -= 0.02
        fig.text(0.08, y, "NHẬN XÉT TỰ ĐỘNG (cần bác sĩ xác nhận)", fontsize=12, weight="bold")
        y -= 0.03
        for line in nhan_xet:
            fig.text(0.10, y, "• " + line, fontsize=9, wrap=True)
            y -= 0.025
        y -= 0.03
        fig.text(0.08, y, "Kết luận của bác sĩ: ...........................................................", fontsize=10)
        fig.text(0.08, 0.04, "Kết quả tự động chỉ dùng cho học tập/nghiên cứu, không thay thế chẩn đoán y khoa.",
                 fontsize=8, style="italic", color="gray")
        pdf.savefig(fig)
        plt.close(fig)
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf


def fmt(v, nd=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


# ----------------------------------------------------------------------------
# 3. GIAO DIỆN - THANH BÊN
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Dữ liệu")
    nguon = st.radio("Nguồn dữ liệu", ["Dùng dữ liệu mẫu", "Tải file lên"])
    files = []
    if nguon == "Tải file lên":
        files = st.file_uploader(
            "File dữ liệu ECG", type=["csv", "txt", "edf", "hea", "dat"] + list(DUOI_NHAN),
            accept_multiple_files=True,
            help="CSV/TXT; EDF/EDF+ (.edf); PhysioNet WFDB: chọn cùng lúc .hea + .dat, "
                 "có thể kèm file nhãn .atr/.qrs/.ann để đối chiếu đỉnh R.") or []

    st.header("2. Thông tin bệnh nhân")
    ten = st.text_input("Họ tên / mã", "Bệnh nhân mẫu")
    tuoi = st.number_input("Tuổi", 1, 120, 25)
    gioi = st.selectbox("Giới tính", ["Nam", "Nữ"])


# ----------------------------------------------------------------------------
# 4. ĐỌC DỮ LIỆU
# ----------------------------------------------------------------------------
if nguon == "Dùng dữ liệu mẫu":
    fs_default = 500
    sig = nk.ecg_simulate(duration=60, sampling_rate=fs_default, heart_rate=75, noise=0.05, random_state=42)
    df = pd.DataFrame({"time_s": np.arange(len(sig)) / fs_default, "ecg_mV": sig})
    meta = dict(fs=None, units={}, ref=None, ky_hieu=None, mo_ta="", nguon_nhan="")
else:
    if not files:
        st.info("⬅️ Hãy tải file ECG lên ở thanh bên: CSV/TXT, EDF (.edf), "
                "hoặc dữ liệu PhysioNet WFDB (.hea + .dat, kèm nhãn .atr/.qrs nếu có).")
        st.stop()
    try:
        df, meta = doc_nhieu_file(files)
    except Exception as e:
        st.error(f"Không đọc được file: {e}")
        st.stop()
    fs_default = int(round(meta["fs"])) if meta["fs"] else 500

with st.sidebar:
    st.header("3. Tín hiệu")
    cols_num = df.select_dtypes(include="number").columns.tolist()
    if not cols_num:
        st.error("File không có cột số.")
        st.stop()
    la_thoi_gian = lambda c: "time" in c.lower() or c.lower() in ("t", "s")
    goi_y = ([c for c in cols_num if any(k in c.lower() for k in ("direct", "ecg", "mlii"))]
             or [c for c in cols_num if not la_thoi_gian(c)] or cols_num)
    col_ecg = st.selectbox("Cột tín hiệu ECG", cols_num, index=cols_num.index(goi_y[0]))
    # Tự ước tính fs nếu có cột thời gian
    col_time = next((c for c in cols_num if la_thoi_gian(c)), None)
    if meta["fs"]:
        col_time = "fs"                     # tần số đã đọc từ file
    elif col_time is not None and len(df) > 2:
        dt = np.median(np.diff(df[col_time].values[:1000]))
        if dt > 0:
            fs_default = int(round(1 / dt))
    fs = st.number_input("Tần số lấy mẫu (Hz)", 50, 10000, fs_default)
    if meta["fs"]:
        st.caption(f"Đọc từ file {meta['mo_ta']}: {meta['fs']:g} Hz, {len(cols_num)} kênh.")
    if col_time is None:
        st.warning("File không có cột thời gian: hãy nhập **đúng tần số lấy mẫu** của thiết bị đo, "
                   "nếu sai thì nhịp tim và mọi khoảng thời gian sẽ sai theo.")

    # Gợi ý đơn vị: dữ liệu toàn số nguyên dương lớn thường là giá trị ADC thô
    v = pd.to_numeric(df[col_ecg], errors="coerce").dropna()
    la_adc = len(v) > 0 and v.min() >= 0 and v.max() > 100 and np.allclose(v, v.round())
    ds_don_vi = ["mV", "V (đổi sang mV)", "µV (đổi sang mV)",
                 "ADC 12-bit → mV (giả định AD8232, 3,3 V, khuếch đại 1100)",
                 "ADC thô (chuẩn hoá, không có đơn vị)"]
    u = meta["units"].get(col_ecg, "").replace("μ", "u").replace("µ", "u").lower()
    chi_so_dv = {"uv": 2, "mv": 0, "v": 1}.get(u, 3 if la_adc else 0)
    don_vi = st.selectbox("Đơn vị tín hiệu", ds_don_vi, index=chi_so_dv,
                          help=f"Đơn vị ghi trong file: {meta['units'][col_ecg]}" if col_ecg in meta["units"] else None)

    goi_y_thai = any(k in " ".join(cols_num + [f.name for f in files]).lower()
                     for k in ("direct", "abdomen", "fetal", "fecg", "adfecg"))
    doi_tuong = st.selectbox("Đối tượng", ["Người lớn", "Thai nhi (FECG)"], index=1 if goi_y_thai else 0,
                             help="Đổi ngưỡng đánh giá: người lớn 60–100 l/p, thai nhi 110–160 l/p (FIGO).")
    thai_nhi = doi_tuong.startswith("Thai")

    dung_sai = None
    if meta["ref"] is not None:
        st.markdown(f"**Nhãn chuẩn:** {len(meta['ref'])} đỉnh R ({meta['nguon_nhan']})")
        dung_sai = st.slider("Dung sai khớp đỉnh R (ms)", 10, 150, 50 if thai_nhi else 150, 10,
                             help="Chuẩn ANSI/AAMI EC57 dùng 150 ms cho người lớn; ECG thai nhi nhịp nhanh "
                                  "nên thường dùng 50 ms.")

    st.header("4. Bộ lọc")
    method = st.selectbox("Phương pháp lọc", ["neurokit", "biosppy", "pantompkins1985", "hamilton2002",
                                              PP_TUY_CHINH])
    cfg = None
    if method == PP_TUY_CHINH:
        nyq = fs / 2
        st.markdown("**Butterworth**")
        bac = st.slider("Bậc lọc", 1, 8, 4,
                        help="Bậc càng cao, sườn cắt càng dốc nhưng dễ gây dao động (ringing) quanh QRS.")
        c_lo, c_hi = st.columns(2)
        lo = c_lo.number_input("Cắt dưới (Hz)", 0.05, 5.0, 0.5, 0.05, format="%.2f",
                               help="0,5 Hz khử trôi nền; 0,05 Hz giữ nguyên đoạn ST (chuẩn chẩn đoán).")
        hi_max = float(np.floor(nyq * 0.99 * 10) / 10)
        hi = c_hi.number_input("Cắt trên (Hz)", 5.0, hi_max, min(40.0, hi_max), 1.0, format="%.1f",
                               help=f"Phải nhỏ hơn tần số Nyquist = fs/2 = {nyq:g} Hz.")
        if lo >= hi:
            st.error("Tần số cắt dưới phải nhỏ hơn tần số cắt trên.")
            st.stop()
        nen = st.radio("Khử trôi đường nền bằng", ["Butterworth thông cao", "Wavelet (bỏ thành phần xấp xỉ)"],
                       help="Wavelet: phân tách tín hiệu, bỏ thành phần tần thấp dưới tần số cắt dưới.")

        st.markdown("**Notch điện lưới**")
        notch = st.checkbox("Bật Notch", True)
        c_f, c_q = st.columns(2)
        f_notch = c_f.selectbox("Tần số", [50, 60], disabled=not notch, help="Việt Nam dùng điện lưới 50 Hz.")
        Q = c_q.number_input("Hệ số Q", 1.0, 100.0, 30.0, 1.0, disabled=not notch,
                             help="Q lớn → khấc cắt hẹp, ít ảnh hưởng tín hiệu lân cận.")
        hai = st.checkbox("Cắt cả hài bậc 2, 3", False, disabled=not notch)

        st.markdown("**Khử nhiễu Wavelet (DWT)**")
        wl_on = st.checkbox("Bật khử nhiễu wavelet", True)
        wavelet = st.selectbox("Wavelet mẹ", ["sym8", "sym4", "db4", "db6", "coif3"],
                               help="sym8 / db4 có hình dạng gần giống phức bộ QRS nên hay dùng cho ECG.")
        muc = st.slider("Số mức phân tách", 1, 8, 4, disabled=not wl_on,
                        help=f"Mức j chứa dải ≈ {fs/2**2:g}–{fs/2:g} Hz ở mức 1, mỗi mức giảm một nửa. "
                             "Mức quá sâu sẽ làm méo QRS.")
        if wl_on and fs / 2 ** (muc + 1) < 5:
            st.warning(f"Mức {muc} chạm tới dải {fs/2**(muc+1):.1f}–{fs/2**muc:.1f} Hz, nơi chứa sóng P và T. "
                       "Ngưỡng hoá ở đây có thể xoá mất P/T; nên giảm số mức.")
        nguong = st.selectbox("Quy tắc ngưỡng", ["Universal (VisuShrink)", "Bayes (BayesShrink)"],
                              disabled=not wl_on)
        c_m, c_t = st.columns(2)
        mem = c_m.radio("Kiểu ngưỡng", ["Mềm (soft)", "Cứng (hard)"], disabled=not wl_on) == "Mềm (soft)"
        theo_muc = c_t.radio("Ước lượng nhiễu", ["Theo từng mức", "Một mức chung"],
                             disabled=not wl_on) == "Theo từng mức"
        cfg = dict(bac=bac, lo=float(lo), hi=float(hi), nen=nen, notch=notch, f_notch=f_notch, Q=float(Q),
                   hai=hai, wl_on=wl_on, wavelet=wavelet, muc=muc, nguong=nguong, mem=mem, theo_muc=theo_muc)
    chay = st.button("▶ Phân tích", type="primary", use_container_width=True)

ecg_raw = pd.to_numeric(df[col_ecg], errors="coerce").interpolate().bfill().values
if don_vi.startswith("V"):
    ecg_raw = ecg_raw * 1000
elif don_vi.startswith("µV"):
    ecg_raw = ecg_raw / 1000
elif don_vi.startswith("ADC 12-bit"):
    # 1 mức ADC = 3,3 V / 4095 tại ngõ vào ADC; chia hệ số khuếch đại 1100 để ra điện thế tại điện cực
    ecg_raw = (ecg_raw - np.median(ecg_raw)) * 3300 / 4095 / 1100
elif don_vi.startswith("ADC thô"):
    ecg_raw = (ecg_raw - np.median(ecg_raw)) / (np.std(ecg_raw) or 1)

duration = len(ecg_raw) / fs
ten_file = ",".join(sorted(f.name for f in files)) if nguon == "Tải file lên" else "du_lieu_mau"
khoa_dau_vao = (ten_file, len(ecg_raw), col_ecg, fs, don_vi, method, tuple(sorted((cfg or {}).items())))
ref = meta["ref"] if meta["ref"] is not None and (not meta["fs"] or fs == fs_default) else None
if st.session_state.get("khoa") != khoa_dau_vao:   # đổi dữ liệu/tham số -> bỏ kết quả cũ
    st.session_state.pop("kq", None)

# ----------------------------------------------------------------------------
# 5. THÔNG TIN FILE
# ----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Số mẫu", f"{len(ecg_raw):,}")
c2.metric("Tần số lấy mẫu", f"{fs} Hz")
c3.metric("Thời lượng", f"{duration:.1f} s")
c4.metric("Kênh phân tích", col_ecg)

with st.expander("Xem 10 dòng đầu của dữ liệu"):
    st.dataframe(df.head(10), use_container_width=True)

if not chay and "kq" not in st.session_state:
    st.info("Bấm **▶ Phân tích** ở thanh bên để bắt đầu.")
    st.stop()

# ----------------------------------------------------------------------------
# 6. PHÂN TÍCH
# ----------------------------------------------------------------------------
if chay or "kq" not in st.session_state:
    with st.spinner("Đang lọc tín hiệu, tìm đỉnh R và tính chỉ số..."):
        try:
            ecg_clean, rpeaks, rr_ms, hr, quality = xu_ly_ecg(ecg_raw, fs, method, cfg)
        except Exception as e:
            st.error(f"Lỗi khi phân tích: {e}")
            st.stop()
        khoang = tinh_khoang_song(ecg_clean, rpeaks, fs)
        hrv = tinh_hrv(rpeaks, fs) if duration >= 30 else {}
    st.session_state.kq = dict(ecg_clean=ecg_clean, rpeaks=rpeaks, rr_ms=rr_ms, hr=hr,
                               quality=quality, khoang=khoang, hrv=hrv)
    st.session_state.khoa = khoa_dau_vao

kq = st.session_state.kq
ecg_clean, rpeaks, rr_ms, hr = kq["ecg_clean"], kq["rpeaks"], kq["rr_ms"], kq["hr"]
khoang, hrv, quality = kq["khoang"], kq["hrv"], kq["quality"]
hr_mean = 60000 / np.mean(rr_ms)
nhan_xet = danh_gia(hr_mean, rr_ms, khoang, thai_nhi)
dc = doi_chieu(rpeaks, ref, fs, dung_sai) if ref is not None and len(ref) else None

# ----------------------------------------------------------------------------
# 7. HIỂN THỊ KẾT QUẢ
# ----------------------------------------------------------------------------
if not ((50 <= hr_mean <= 240) if thai_nhi else (35 <= hr_mean <= 200)):
    st.warning(f"Nhịp tim tính ra {hr_mean:.0f} l/p là bất thường về mặt sinh lý. "
               "Nhiều khả năng **tần số lấy mẫu đang nhập chưa đúng**, hãy kiểm tra lại.")
if don_vi.startswith("ADC 12-bit"):
    st.caption("Biên độ mV được quy đổi từ giá trị ADC theo giả định AD8232 (3,3 V, khuếch đại 1100), "
               "chỉ mang tính ước lượng.")
st.subheader("Kết quả chính")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nhịp tim TB", f"{hr_mean:.0f} l/p")
m2.metric("Nhịp tim min", f"{hr.min():.0f} l/p")
m3.metric("Nhịp tim max", f"{hr.max():.0f} l/p")
m4.metric("Số nhịp", f"{len(rpeaks)}")
m5.metric("Chất lượng tín hiệu", "—" if not np.isfinite(quality) else f"{quality*100:.0f}%")

if dc is not None:
    st.subheader("🎯 Đối chiếu với nhãn chuẩn")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Độ nhạy Se", f"{dc['Se']*100:.1f}%", help="Tỷ lệ nhãn chuẩn được app tìm thấy = TP/(TP+FN)")
    n2.metric("Giá trị dự báo dương PPV", f"{dc['PPV']*100:.1f}%", help="Tỷ lệ đỉnh app tìm là đúng = TP/(TP+FP)")
    n3.metric("F1", f"{dc['F1']*100:.1f}%")
    n4.metric("Lệch thời gian TB", f"{dc['lech_ms']:.1f} ms")
    st.caption(f"TP = {dc['TP']} · bỏ sót FN = {dc['FN']} · phát hiện thừa FP = {dc['FP']} · "
               f"dung sai ±{dung_sai} ms · nhãn: {meta['nguon_nhan']}.")
    if thai_nhi and "abdomen" in col_ecg.lower():
        st.info("Kênh bụng mẹ (Abdomen) chứa cả ECG của mẹ (biên độ lớn) lẫn của thai nhi, nên thuật toán chủ yếu "
                "bắt nhịp **mẹ**. Nhãn chuẩn là nhịp **thai**, vì vậy Se/PPV thấp là đúng bản chất tín hiệu. "
                "Hãy chọn kênh **Direct_1** (điện cực đặt trên đầu thai nhi) để đánh giá thuật toán.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Dải sóng ECG", "🧹 Trước/sau lọc", "❤️ Nhịp tim & HRV", "📋 Bảng chỉ số", "📄 Báo cáo"]
)

with tab1:
    max_start = max(0.0, duration - 10)
    t_start = st.slider("Bắt đầu từ (s)", 0.0, float(max_start), 0.0, 0.5)
    dur = st.select_slider("Độ dài đoạn hiển thị (s)", [5, 10], value=10)
    fig_strip = fig_dai_song(ecg_clean, rpeaks, fs, t_start, min(dur, duration), ref)
    st.pyplot(fig_strip)

with tab2:
    fig_loc = fig_truoc_sau_loc(ecg_raw, ecg_clean, fs)
    st.pyplot(fig_loc)
    st.caption(f"Phương pháp lọc: {mo_ta_loc(method, cfg)}")
    fig_pho_ = fig_pho(ecg_raw, ecg_clean, fs)
    st.pyplot(fig_pho_)
    fig_dap_ung_ = None
    if method == PP_TUY_CHINH:
        fig_dap_ung_ = fig_dap_ung(fs, cfg)
        st.pyplot(fig_dap_ung_)
        st.caption("Đáp ứng trên chỉ gồm phần lọc tuyến tính; khử nhiễu wavelet là phi tuyến nên không có "
                   "đáp ứng tần số cố định, xem hiệu quả của nó trên phổ công suất.")

with tab3:
    fig_hr = fig_nhip_tim(rpeaks, hr, fs, nguong_nhip(thai_nhi))
    st.pyplot(fig_hr)
    fig_rr_ = fig_rr(rr_ms)
    st.pyplot(fig_rr_)
    if not hrv:
        st.warning("Tín hiệu ngắn hơn 30 s nên không tính HRV. Nên ghi tối thiểu 5 phút cho HRV ngắn hạn.")

# Bảng chỉ số tổng hợp
rows = [
    ("Nhịp tim trung bình (l/p)", fmt(hr_mean, 0)),
    ("Nhịp tim min / max (l/p)", f"{hr.min():.0f} / {hr.max():.0f}"),
    ("RR trung bình (ms)", fmt(np.mean(rr_ms), 0)),
]
rows += [(k, fmt(v, 0)) for k, v in khoang.items()]
rows += [(k, fmt(v, 1)) for k, v in hrv.items()]
if dc is not None:
    rows += [("Đối chiếu nhãn chuẩn: Se (%)", fmt(dc["Se"] * 100, 1)),
             ("Đối chiếu nhãn chuẩn: PPV (%)", fmt(dc["PPV"] * 100, 1)),
             ("Đối chiếu nhãn chuẩn: F1 (%)", fmt(dc["F1"] * 100, 1)),
             ("Đối chiếu nhãn chuẩn: TP / FN / FP", f"{dc['TP']} / {dc['FN']} / {dc['FP']}")]
bang = pd.DataFrame(rows, columns=["Chỉ số", "Giá trị"])

with tab4:
    st.dataframe(bang, use_container_width=True, hide_index=True)
    st.markdown("**Nhận xét tự động** (chỉ mang tính tham khảo):")
    for line in nhan_xet:
        st.markdown(f"- {line}")

    df_beats = pd.DataFrame({
        "Nhịp số": np.arange(1, len(rr_ms) + 1),
        "Thời điểm R (s)": np.round(rpeaks[1:] / fs, 3),
        "RR (ms)": np.round(rr_ms, 1),
        "Nhịp tim (l/p)": np.round(hr, 1),
    })
    with st.expander("Bảng từng nhịp"):
        st.dataframe(df_beats, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇ Tải bảng chỉ số (CSV)", bang.to_csv(index=False).encode("utf-8-sig"),
                       "chi_so_ecg.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇ Tải bảng từng nhịp (CSV)", df_beats.to_csv(index=False).encode("utf-8-sig"),
                       "tung_nhip_ecg.csv", "text/csv", use_container_width=True)

with tab5:
    thong_tin = {
        "Bệnh nhân": ten, "Tuổi": tuoi, "Giới tính": gioi,
        "Ngày phân tích": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Đối tượng": doi_tuong, "Dữ liệu": f"{ten_file} · kênh {col_ecg}",
        "Tần số lấy mẫu": f"{fs} Hz", "Thời lượng ghi": f"{duration:.1f} s",
        "Phương pháp lọc": mo_ta_loc(method, cfg),
    }
    st.write("Báo cáo gồm thông tin bệnh nhân, bảng chỉ số, nhận xét tự động, dải sóng ECG và các biểu đồ.")
    figs_pdf = [fig_strip, fig_loc, fig_pho_] + ([fig_dap_ung_] if fig_dap_ung_ is not None else []) + [fig_hr, fig_rr_]
    pdf = tao_pdf(thong_tin, bang, nhan_xet, figs_pdf)
    st.download_button("⬇ Tải báo cáo PDF", pdf, f"bao_cao_ecg_{ten.replace(' ', '_')}.pdf",
                       "application/pdf", type="primary", use_container_width=True)

plt.close("all")
