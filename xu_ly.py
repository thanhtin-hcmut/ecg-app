"""Đọc dữ liệu, lọc tín hiệu và tính chỉ số ECG (không phụ thuộc giao diện)."""
import io

import neurokit2 as nk
import numpy as np
import pandas as pd

from doc_physionet import doc_edf, doc_nhan_wfdb, doc_wfdb
from i18n import t


# ----------------------------------------------------------------------------
# Đọc file
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
        raise ValueError(t("File rỗng."))
    mau = dong[1] if len(dong) > 1 else dong[0]
    sep = next((d for d in [";", "\t", ","] if d in mau), r"\s+")
    tach = lambda l: [x for x in (l.split(sep) if sep != r"\s+" else l.split()) if x.strip() != ""]
    co_tieu_de = not all(_la_so(x.strip()) for x in tach(dong[0]))
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=0 if co_tieu_de else None)
    if not co_tieu_de:
        df.columns = ["ecg"] if df.shape[1] == 1 else [f"kenh_{i+1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df.dropna(axis=1, how="all")


def doc_acq(b: bytes) -> dict:
    """File Biopac AcqKnowledge / Biopac Student Lab (.acq; .gtl chỉ là mẫu, thường không có dữ liệu)."""
    try:
        import bioread
    except ImportError as e:
        raise ValueError(t("Cần cài thư viện bioread để đọc file Biopac: pip install bioread")) from e
    d = bioread.read(io.BytesIO(b))
    out = dict(signals={}, units={}, fs={}, ann_s=[])
    for i, c in enumerate(d.channels):
        if c.data is None or len(c.data) == 0:
            continue
        ten = (c.name or f"CH{i+1}").strip()
        if ten in out["signals"]:
            ten = f"{ten} ({i+1})"
        out["signals"][ten] = np.asarray(c.data, dtype=float)
        out["units"][ten] = (c.units or "").strip()
        out["fs"][ten] = float(c.samples_per_second)
    if not out["signals"]:
        raise ValueError(t("File Biopac này không có dữ liệu (file .gtl chỉ là mẫu cài đặt bài thí nghiệm). "
                           "Hãy mở bản ghi trong Biopac Student Lab rồi lưu thành file .acq."))
    return out


DUOI_NHAN = ("atr", "qrs", "ann")
DUOI_HO_TRO = ["csv", "txt", "edf", "hea", "dat", "acq", "gtl"] + list(DUOI_NHAN)
_KENH_ECG = ("ecg", "ekg", "mlii", "direct", "lead")


def doc_nhieu_file(files) -> tuple[pd.DataFrame, dict]:
    """Đọc 1 hoặc nhiều file: CSV/TXT, EDF/EDF+, WFDB (.hea + .dat), Biopac (.acq) và nhãn WFDB.
    Trả về (bảng tín hiệu, thông tin: fs, đơn vị từng kênh, đỉnh R chuẩn nếu có, mô tả định dạng)."""
    duoi = lambda f: f.name.lower().rsplit(".", 1)[-1]
    lay = lambda *d: [f for f in files if duoi(f) in d]
    meta = dict(fs=None, units={}, ref=None, ky_hieu=None, mo_ta="", nguon_nhan="")

    if lay("edf"):
        f = lay("edf")[0]
        d, meta["mo_ta"] = doc_edf(f.getvalue()), f"EDF/EDF+ ({f.name})"
    elif lay("acq", "gtl"):
        f = lay("acq", "gtl")[0]
        d, meta["mo_ta"] = doc_acq(f.getvalue()), f"Biopac ({f.name})"
    elif lay("hea"):
        f = lay("hea")[0]
        dat = {g.name: g.getvalue() for g in lay("dat")}
        d, meta["mo_ta"] = doc_wfdb(f.getvalue().decode("latin-1"), dat), f"WFDB ({f.name})"
    elif lay("csv", "txt"):
        return doc_file(lay("csv", "txt")[0]), meta
    elif lay("dat"):
        raise ValueError(t("File .dat của PhysioNet cần tải lên kèm file .hea cùng tên."))
    else:
        raise ValueError(t("Chưa có file tín hiệu. Hãy tải .csv/.txt, .edf, .acq hoặc cặp .hea + .dat."))

    if not d["signals"]:
        raise ValueError(t("File không có kênh tín hiệu nào."))
    # giữ các kênh cùng tần số lấy mẫu: ưu tiên tần số của kênh ECG, nếu không có thì tần số phổ biến nhất
    fs_ecg = [d["fs"][k] for k in d["signals"] if any(s in k.lower() for s in _KENH_ECG)]
    fs_ds = list(d["fs"].values())
    fs = fs_ecg[0] if fs_ecg else max(set(fs_ds), key=fs_ds.count)
    ten = [k for k in d["signals"] if d["fs"][k] == fs]
    n = min(len(d["signals"][k]) for k in ten)
    df = pd.DataFrame({k: d["signals"][k][:n] for k in ten})
    meta.update(fs=fs, units={k: d["units"][k] for k in ten}, bo_kenh=len(d["signals"]) - len(ten))

    nhan = lay(*DUOI_NHAN)
    if nhan:
        mau, ky = doc_nhan_wfdb(nhan[0].getvalue())
        meta.update(ref=mau[mau < n], ky_hieu=ky, nguon_nhan=nhan[0].name)
    elif d["ann_s"]:
        mau = np.round(np.asarray(d["ann_s"]) * fs).astype(int)
        meta.update(ref=mau[(mau >= 0) & (mau < n)], nguon_nhan=t("nhãn EDF+ trong file"))
    return df, meta


# ----------------------------------------------------------------------------
# Thiết bị đo và đơn vị
# ----------------------------------------------------------------------------
DS_DON_VI = ["mV", "V (đổi sang mV)", "µV (đổi sang mV)",
             "ADC 12-bit → mV (giả định AD8232, 3,3 V, khuếch đại 1100)",
             "ADC 10-bit → mV (giả định AD8232, 5 V, khuếch đại 1100)",
             "ADC thô (chuẩn hoá, không có đơn vị)"]
KHONG_CHON = "Không chọn (tự nhập)"
# tên: (tần số lấy mẫu thường dùng, đơn vị, ghi chú)
THIET_BI = {
    KHONG_CHON: None,
    "Biopac MP36 / MP45 (Student Lab)": (1000, DS_DON_VI[0], "BSL Lesson ECG mặc định 1000 Hz; file .acq đã ghi sẵn tần số."),
    "Biopac MP150 / MP160 (AcqKnowledge)": (1000, DS_DON_VI[0], "Thường ghi 500–2000 Hz; file .acq đã ghi sẵn tần số."),
    "AD8232 + ESP32 (ADC 12-bit, 3,3 V)": (250, DS_DON_VI[3], "Tần số phụ thuộc lệnh delay trong code: delay(4) ≈ 250 Hz, delay(5) ≈ 200 Hz."),
    "AD8232 + Arduino Uno/Nano (ADC 10-bit, 5 V)": (250, DS_DON_VI[4], "Tần số phụ thuộc lệnh delay trong code: delay(4) ≈ 250 Hz, delay(5) ≈ 200 Hz."),
    "MIT-BIH / Holter (PhysioNet)": (360, DS_DON_VI[0], "MIT-BIH Arrhythmia ghi 360 Hz."),
    "Máy điện tim 12 chuyển đạo (PTB-XL)": (500, DS_DON_VI[0], "PTB-XL có bản 500 Hz và 100 Hz."),
    "Đai ngực Polar H10": (130, DS_DON_VI[2], "Polar H10 xuất ECG 130 Hz, đơn vị µV."),
}


def doi_don_vi(x: np.ndarray, don_vi: str) -> np.ndarray:
    if don_vi.startswith("V"):
        return x * 1000
    if don_vi.startswith("µV"):
        return x / 1000
    if don_vi.startswith("ADC 12-bit"):      # 1 mức = 3,3 V / 4095 tại ngõ vào ADC, chia khuếch đại 1100
        return (x - np.median(x)) * 3300 / 4095 / 1100
    if don_vi.startswith("ADC 10-bit"):      # 1 mức = 5 V / 1023
        return (x - np.median(x)) * 5000 / 1023 / 1100
    if don_vi.startswith("ADC thô"):
        return (x - np.median(x)) / (np.std(x) or 1)
    return x


def chi_so_don_vi_mac_dinh(don_vi_file: str, v: pd.Series) -> int:
    u = (don_vi_file or "").replace("μ", "u").replace("µ", "u").lower()
    if u in ("mv", "millivolts"):
        return 0
    if u in ("v", "volts"):
        return 1
    if u in ("uv", "microvolts"):
        return 2
    la_adc = len(v) > 0 and v.min() >= 0 and v.max() > 100 and np.allclose(v, v.round())
    if la_adc:
        return 4 if v.max() <= 1023 else 3
    return 0


# ----------------------------------------------------------------------------
# Đối chiếu nhãn chuẩn
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Bộ lọc
# ----------------------------------------------------------------------------
PP_TUY_CHINH = "Tùy chỉnh (Butterworth + Notch + Wavelet)"
DS_PHUONG_PHAP = ["neurokit", "biosppy", "pantompkins1985", "hamilton2002", PP_TUY_CHINH]
DS_NEN = ["Butterworth thông cao", "Wavelet (bỏ thành phần xấp xỉ)"]
DS_WAVELET = ["sym8", "sym4", "db4", "db6", "coif3"]
DS_NGUONG = ["Universal (VisuShrink)", "Bayes (BayesShrink)"]
DS_KIEU_NGUONG = ["Mềm (soft)", "Cứng (hard)"]
DS_UOC_LUONG = ["Theo từng mức", "Một mức chung"]
TEN_PP = {"neurokit": "Tiêu chuẩn (mặc định)", "biosppy": "Giữ hình dạng sóng",
          "pantompkins1985": "Đếm nhịp – gọn nhẹ", "hamilton2002": "Đếm nhịp – ít nhiễu chen",
          PP_TUY_CHINH: "Tự chỉnh"}
# (phù hợp khi nào, lưu ý, tên thuật toán)
GIOI_THIEU_PP = {
    "neurokit": ("Ổn định, dùng tốt cho hầu hết bản ghi.", "Đỉnh R thấp đi một chút sau khi lọc.", "NeuroKit2"),
    "biosppy": ("Sóng P, QRS, T rõ nét nhất. Nên chọn khi cần đo PR, QRS, QT.",
                "Bản ghi nhiều nhiễu cơ thì đường sóng còn hơi gợn.", "BioSPPy"),
    "pantompkins1985": ("Thuật toán kinh điển, gọn nhẹ. Hợp để đếm nhịp và tính nhịp tim.",
                        "Làm mất sóng P, T: không dùng để đo PR, QT.", "Pan-Tompkins 1985"),
    "hamilton2002": ("Dải lọc hẹp hơn nên ít nhiễu chen vào. Hợp để đếm nhịp.",
                     "Làm mất sóng P, T: không dùng để đo PR, QT.", "Hamilton 2002"),
    PP_TUY_CHINH: ("Tự chọn từng bước lọc, dành cho người cần chỉnh chi tiết.",
                   "Chỉnh không hợp lý có thể làm méo sóng. Nên bấm Đề xuất trước.", "Butterworth + Notch + Wavelet"),
}
TEN_NEN = {DS_NEN[0]: "Bộ lọc thông thường", DS_NEN[1]: "Wavelet (tốt khi nền trôi mạnh)"}
TEN_NGUONG = {DS_NGUONG[0]: "Mạnh (Universal)", DS_NGUONG[1]: "Nhẹ, giữ chi tiết (Bayes)"}
TEN_KIEU = {DS_KIEU_NGUONG[0]: "Mềm (mượt hơn)", DS_KIEU_NGUONG[1]: "Cứng (giữ biên độ)"}
TEN_UOC = {DS_UOC_LUONG[0]: "Từng mức", DS_UOC_LUONG[1]: "Một mức chung"}
MAC_DINH_LOC = dict(method="neurokit", cf_bac=4, cf_lo=0.5, cf_hi=40.0, cf_nen=DS_NEN[0], cf_notch=True,
                    cf_fnotch=50, cf_Q=30.0, cf_hai=False, cf_wl=True, cf_wavelet="sym8", cf_muc=4,
                    cf_nguong=DS_NGUONG[0], cf_mem=DS_KIEU_NGUONG[0], cf_theomuc=DS_UOC_LUONG[0])


def ten_phuong_an(r: dict) -> str:
    return t("Tự chỉnh riêng cho file này") if r["method"] == PP_TUY_CHINH else t(TEN_PP.get(r["method"], r["ten"]))


def cfg_tu_session(ss) -> dict:
    """Cấu hình lọc tùy chỉnh lấy từ session_state (dùng cả khi các ô chỉnh đang ẩn)."""
    return dict(bac=int(ss["cf_bac"]), lo=float(ss["cf_lo"]), hi=float(ss["cf_hi"]), nen=ss["cf_nen"],
                notch=bool(ss["cf_notch"]), f_notch=int(ss["cf_fnotch"]), Q=float(ss["cf_Q"]), hai=bool(ss["cf_hai"]),
                wl_on=bool(ss["cf_wl"]), wavelet=ss["cf_wavelet"], muc=int(ss["cf_muc"]), nguong=ss["cf_nguong"],
                mem=ss["cf_mem"] == DS_KIEU_NGUONG[0], theo_muc=ss["cf_theomuc"] == DS_UOC_LUONG[0])


def cfg_sang_session(cfg: dict) -> dict:
    return dict(cf_bac=int(cfg["bac"]), cf_lo=float(cfg["lo"]), cf_hi=float(cfg["hi"]), cf_nen=cfg["nen"],
                cf_notch=bool(cfg["notch"]), cf_fnotch=int(cfg["f_notch"]), cf_Q=float(cfg["Q"]),
                cf_hai=bool(cfg["hai"]), cf_wl=bool(cfg["wl_on"]), cf_wavelet=cfg["wavelet"],
                cf_muc=int(cfg["muc"]), cf_nguong=cfg["nguong"],
                cf_mem=DS_KIEU_NGUONG[0] if cfg["mem"] else DS_KIEU_NGUONG[1],
                cf_theomuc=DS_UOC_LUONG[0] if cfg["theo_muc"] else DS_UOC_LUONG[1])


def _nguong_wavelet(d, sigma, kieu):
    """Ngưỡng cho một mức chi tiết: Universal (VisuShrink) hoặc BayesShrink."""
    if kieu.startswith("Universal"):
        return sigma * np.sqrt(2 * np.log(len(d)))
    sigma_x = np.sqrt(max(np.var(d) - sigma ** 2, 1e-12))
    return sigma ** 2 / sigma_x


def loc_tuy_chinh(x: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Chuỗi lọc tùy chỉnh (không lệch pha): 1) khử trôi nền (Butterworth thông cao hoặc wavelet),
    2) Butterworth thông dải/thông thấp, 3) Notch 50/60 Hz (+ hài), 4) khử nhiễu wavelet."""
    import pywt
    from scipy import signal as ss

    y = np.asarray(x, dtype=float) - np.median(x)
    nyq = fs / 2
    lo, hi = cfg["lo"], min(cfg["hi"], 0.99 * nyq)
    if cfg["nen"].startswith("Wavelet"):
        w = pywt.Wavelet(cfg["wavelet"])
        L = int(np.ceil(np.log2(fs / lo))) - 1          # dải xấp xỉ [0, fs/2^(L+1)] nằm dưới tần số cắt dưới
        L = int(np.clip(L, 1, pywt.dwt_max_level(len(y), w.dec_len)))
        c = pywt.wavedec(y, w, level=L)
        c[0] = np.zeros_like(c[0])
        y = pywt.waverec(c, w)[: len(x)]
        sos = ss.butter(cfg["bac"], hi, btype="lowpass", fs=fs, output="sos")
    else:
        sos = ss.butter(cfg["bac"], [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = ss.sosfiltfilt(sos, y)
    if cfg["notch"]:
        for k in ([1, 2, 3] if cfg["hai"] else [1]):
            f0 = cfg["f_notch"] * k
            if f0 < 0.99 * nyq:
                b, a = ss.iirnotch(f0, cfg["Q"], fs=fs)
                y = ss.filtfilt(b, a, y)
    if cfg["wl_on"]:
        w = pywt.Wavelet(cfg["wavelet"])
        L = min(cfg["muc"], pywt.dwt_max_level(len(y), w.dec_len))
        c = pywt.wavedec(y, w, level=L)
        sigma1 = np.median(np.abs(c[-1])) / 0.6745
        for i in range(1, len(c)):
            sigma = np.median(np.abs(c[i])) / 0.6745 if cfg["theo_muc"] else sigma1
            c[i] = pywt.threshold(c[i], _nguong_wavelet(c[i], sigma, cfg["nguong"]),
                                  mode="soft" if cfg["mem"] else "hard")
        y = pywt.waverec(c, w)[: len(x)]
    return y


def mo_ta_loc(method: str, cfg: dict) -> str:
    """Mô tả ngắn gọn, dễ hiểu bộ lọc đang dùng."""
    if method != PP_TUY_CHINH:
        return f"{t(TEN_PP.get(method, method))} ({GIOI_THIEU_PP.get(method, ('', '', method))[2]})"
    s = [t("giữ dải {lo}–{hi} Hz", lo=f"{cfg['lo']:g}", hi=f"{cfg['hi']:g}")]
    if cfg["nen"].startswith("Wavelet"):
        s.append(t("làm phẳng nền bằng wavelet"))
    if cfg["notch"]:
        s.append(t("lọc nhiễu điện {f} Hz", f=cfg["f_notch"]))
    if cfg["wl_on"]:
        s.append(t("làm mịn wavelet"))
    return t("Tự chỉnh") + ": " + ", ".join(s)


# ----------------------------------------------------------------------------
# Phân tích
# ----------------------------------------------------------------------------
def xu_ly_ecg(ecg_raw: np.ndarray, fs: int, method: str, cfg: dict | None = None):
    """Lọc tín hiệu, tìm đỉnh R, tính nhịp tim và chất lượng tín hiệu."""
    if method == PP_TUY_CHINH:
        ecg_clean = loc_tuy_chinh(ecg_raw, fs, cfg)
    else:
        ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=fs, method=method)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, correct_artifacts=True)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(rpeaks) < 3:
        raise ValueError(t("Tìm được quá ít đỉnh R. Kiểm tra lại cột dữ liệu và tần số lấy mẫu."))
    rr_ms = np.diff(rpeaks) / fs * 1000
    hr = 60000 / rr_ms
    try:
        quality = float(np.nanmean(nk.ecg_quality(ecg_clean, rpeaks=rpeaks, sampling_rate=fs)))
    except Exception:
        quality = np.nan
    return ecg_clean, rpeaks, rr_ms, hr, quality


def tinh_khoang_song(ecg_clean, rpeaks, fs):
    """PR, QRS, QT, QTc (Bazett, Fridericia) từ hình thái sóng; dùng tối đa 300 nhịp đầu."""
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

    pr = trung_vi("ECG_P_Onsets", "ECG_R_Onsets")
    qrs = trung_vi("ECG_Q_Peaks", "ECG_S_Peaks")          # đỉnh Q/S ổn định hơn mốc onset/offset của DWT
    qt = trung_vi("ECG_Q_Peaks", "ECG_T_Offsets")
    rr_s = np.median(np.diff(rpeaks)) / fs
    qtc_b = qt / np.sqrt(rr_s) if np.isfinite(qt) else np.nan
    qtc_f = qt / np.cbrt(rr_s) if np.isfinite(qt) else np.nan
    return {"PR (ms)": pr, "QRS (ms)": qrs, "QT (ms)": qt, "QTc Bazett (ms)": qtc_b, "QTc Fridericia (ms)": qtc_f}


def tinh_hrv(rpeaks, fs):
    """HRV miền thời gian và Poincaré (SD1 = √0,5·SD(ΔRR), SD2 = √(2·SDNN² − SD1²))."""
    hrv = nk.hrv_time(rpeaks, sampling_rate=fs)
    lay = lambda c: float(hrv[c].iloc[0]) if c in hrv else np.nan
    rr = np.diff(rpeaks) / fs * 1000
    sd1 = np.sqrt(0.5) * np.std(np.diff(rr), ddof=1)
    sd2 = np.sqrt(max(2 * np.var(rr, ddof=1) - sd1 ** 2, 0))
    return {"MeanNN (ms)": lay("HRV_MeanNN"), "SDNN (ms)": lay("HRV_SDNN"), "RMSSD (ms)": lay("HRV_RMSSD"),
            "pNN50 (%)": lay("HRV_pNN50"), "SD1 (ms)": float(sd1), "SD2 (ms)": float(sd2)}


def nguong_nhip(thai_nhi: bool):
    """Người lớn khi nghỉ 60–100 l/p; thai nhi (FIGO) 110–160 l/p."""
    return (110, 160) if thai_nhi else (60, 100)


def fmt(v, nd=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
