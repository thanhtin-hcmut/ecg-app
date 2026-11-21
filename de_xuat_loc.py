"""
Đề xuất bộ lọc cho từng file ECG.

Quy trình:
  1. Phân tích phổ tín hiệu gốc (Welch) để đo: trôi đường nền, nhiễu điện lưới 50/60 Hz,
     nhiễu tần cao  →  tự sinh một cấu hình "Tùy chỉnh" riêng cho file.
  2. Chạy thử mọi phương pháp lọc (4 thuật toán có sẵn + cấu hình tự sinh) trên cùng một đoạn tín hiệu.
  3. Chấm điểm mỗi phương pháp theo 4 tiêu chí khách quan (0–1):
       - Chất lượng tín hiệu (SQI): độ giống nhau giữa từng nhịp và nhịp trung bình (NeuroKit)
       - Độ hợp lý của nhịp: tỷ lệ khoảng RR nằm trong ngưỡng sinh lý và không nhảy bất thường
       - Giữ hình dạng sóng: tương quan nhịp trung bình sau lọc với nhịp trung bình của tín hiệu
         chỉ lọc nhẹ (giữ P, QRS, T)
       - Khử nhiễu: tỷ lệ công suất nằm trong dải hữu ích của ECG sau lọc
     Nếu có nhãn đỉnh R chuẩn: điểm = F1 so với nhãn (tiêu chí khách quan nhất).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss

_tp = getattr(np, "trapezoid", None) or np.trapz
TRONG_SO = {"sqi": 0.35, "rr": 0.25, "hinh_dang": 0.25, "khu_nhieu": 0.15}


# ----------------------------------------------------------------------------
# 1. Phân tích phổ → cấu hình tùy chỉnh riêng cho file
# ----------------------------------------------------------------------------
def _cong_suat(f, p, a, b):
    m = (f >= a) & (f < b)
    return float(_tp(p[m], f[m])) if m.any() else 0.0


def phan_tich_pho(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, float) - np.mean(x)
    n = int(min(len(x), 8 * fs))
    f, p = ss.welch(x, fs=fs, nperseg=n)
    nyq = fs / 2
    tong = _cong_suat(f, p, 0.05, nyq) or 1e-20

    # Nhiễu điện lưới: so mật độ công suất tại f0 với nền xung quanh
    def do_noi(f0):
        if f0 + 6 >= nyq:
            return 0.0
        dinh = p[(f >= f0 - 0.6) & (f <= f0 + 0.6)].max()
        nen = np.median(p[((f >= f0 - 6) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 6))])
        return float(dinh / (nen + 1e-30))

    luoi = {50: do_noi(50), 60: do_noi(60)}
    f_luoi = max(luoi, key=luoi.get)
    return dict(
        nen=_cong_suat(f, p, 0.05, 0.5) / tong,           # tỷ lệ công suất trôi nền
        tan_cao=_cong_suat(f, p, 40, nyq) / tong,         # tỷ lệ công suất tần cao (nhiễu cơ, nhiễu điện tử)
        f_luoi=f_luoi, do_noi_luoi=luoi[f_luoi],
        hai_luoi=do_noi(2 * f_luoi) if 2 * f_luoi + 6 < nyq else 0.0,
    )


def cau_hinh_tu_dong(pho: dict, fs: float, thai_nhi: bool) -> tuple[dict, list[str]]:
    """Sinh cấu hình Butterworth + Notch + Wavelet phù hợp với phổ của file, kèm lý do."""
    nyq = fs / 2
    ly_do = []
    # Dải thông: ECG thai nhi có QRS hẹp, nhiều năng lượng tần cao hơn người lớn
    hi = min(70.0 if thai_nhi else 40.0, round(0.45 * fs))
    lo = 1.0 if thai_nhi else 0.5
    ly_do.append(f"Dải thông {lo:g}–{hi:g} Hz: " + ("QRS thai nhi hẹp, cần giữ thành phần tần cao hơn."
                                                     if thai_nhi else "dải chuẩn cho theo dõi ECG người lớn."))

    nen = "Wavelet (bỏ thành phần xấp xỉ)" if pho["nen"] > 0.3 else "Butterworth thông cao"
    ly_do.append(f"Trôi đường nền chiếm {pho['nen']*100:.0f}% công suất → "
                 + ("khử nền bằng wavelet (bám nền tốt hơn khi trôi mạnh)." if nen.startswith("Wavelet")
                    else "Butterworth thông cao là đủ."))

    notch = pho["do_noi_luoi"] > 3 and pho["f_luoi"] < 0.99 * nyq
    hai = notch and pho["hai_luoi"] > 3
    if notch:
        ly_do.append(f"Có đỉnh nhiễu điện lưới {pho['f_luoi']} Hz cao gấp {pho['do_noi_luoi']:.0f} lần nền → bật Notch"
                     + (" kèm hài." if hai else "."))
    else:
        ly_do.append("Không thấy đỉnh nhiễu điện lưới rõ → tắt Notch để không cắt thêm tín hiệu.")

    wl_on = pho["tan_cao"] > 0.02
    # Mức wavelet sâu nhất vẫn nằm trên ~8 Hz để không xoá sóng P/T
    muc = int(np.clip(np.floor(np.log2(fs / 8)) - 1, 1, 6))
    ly_do.append(f"Nhiễu tần cao (>40 Hz) chiếm {pho['tan_cao']*100:.1f}% công suất → "
                 + (f"bật khử nhiễu wavelet sym8, {muc} mức (dải sâu nhất ≥ {fs/2**(muc+1):.0f} Hz)."
                    if wl_on else "không cần khử nhiễu wavelet."))

    cfg = dict(bac=4, lo=float(lo), hi=float(hi), nen=nen, notch=bool(notch), f_notch=int(pho["f_luoi"]),
               Q=30.0, hai=bool(hai), wl_on=bool(wl_on), wavelet="sym8", muc=muc,
               nguong="Universal (VisuShrink)", mem=True, theo_muc=True)
    return cfg, ly_do


# ----------------------------------------------------------------------------
# 2. Các tiêu chí chấm điểm
# ----------------------------------------------------------------------------
def _diem_rr(rpeaks, fs, thai_nhi):
    if len(rpeaks) < 4:
        return 0.0
    rr = np.diff(rpeaks) / fs * 1000
    lo, hi = (250, 750) if thai_nhi else (300, 2000)           # 80–240 l/p thai nhi; 30–200 l/p người lớn
    hop_ly = np.mean((rr >= lo) & (rr <= hi))
    nhay = np.mean(np.abs(np.diff(rr)) / rr[:-1] > 0.25) if len(rr) > 2 else 0
    return float(hop_ly * (1 - nhay))


def _nhip_trung_binh(x, rpeaks, a, b):
    ok = rpeaks[(rpeaks - a >= 0) & (rpeaks + b < len(x))]
    if len(ok) < 3:
        return None
    return np.mean([x[r - a:r + b] for r in ok], axis=0)


def _diem_hinh_dang(x_loc, x_nhe, rpeaks, fs, thai_nhi):
    a, b = (int(0.10 * fs), int(0.20 * fs)) if thai_nhi else (int(0.25 * fs), int(0.45 * fs))
    t1 = _nhip_trung_binh(x_loc, rpeaks, a, b)
    t2 = _nhip_trung_binh(x_nhe, rpeaks, a, b)
    if t1 is None or t2 is None or np.std(t1) == 0 or np.std(t2) == 0:
        return 0.0
    return float(max(np.corrcoef(t1, t2)[0, 1], 0))


def _diem_khu_nhieu(x_loc, fs, thai_nhi):
    f, p = ss.welch(x_loc - np.mean(x_loc), fs=fs, nperseg=int(min(len(x_loc), 4 * fs)))
    tong = _cong_suat(f, p, 0.05, fs / 2) or 1e-20
    tren = min(80 if thai_nhi else 40, 0.45 * fs)
    return float(_cong_suat(f, p, 0.5, tren) / tong)


def doi_chieu_f1(det, ref, fs, dung_sai_ms):
    tol = dung_sai_ms * fs / 1000
    det = np.sort(np.asarray(det))
    dung = np.zeros(len(det), bool)
    tp = 0
    for r in np.sort(np.asarray(ref)):
        j = np.searchsorted(det, r)
        uv = [k for k in (j - 1, j) if 0 <= k < len(det) and not dung[k] and abs(det[k] - r) <= tol]
        if uv:
            dung[min(uv, key=lambda k: abs(det[k] - r))] = True
            tp += 1
    se = tp / len(ref) if len(ref) else 0
    ppv = tp / len(det) if len(det) else 0
    return 2 * se * ppv / (se + ppv) if se + ppv else 0.0


# ----------------------------------------------------------------------------
# 3. Chạy thử & xếp hạng
# ----------------------------------------------------------------------------
def de_xuat(x, fs, thai_nhi, xu_ly_ecg, pp_tuy_chinh, ref=None, dung_sai_ms=50, toi_da_s=120):
    """
    x, fs        : tín hiệu (mV) và tần số lấy mẫu
    xu_ly_ecg    : hàm (x, fs, method, cfg) -> (clean, rpeaks, rr_ms, hr, quality) của app
    ref          : vị trí đỉnh R chuẩn (mẫu) nếu có
    Trả về: dict(bang=list kết quả đã xếp hạng, cfg_tu_dong, ly_do, pho, doan_s, co_nhan)
    """
    n = int(min(len(x), toi_da_s * fs))
    x = np.asarray(x[:n], float)
    ref_doan = None if ref is None else np.asarray(ref)[np.asarray(ref) < n]

    pho = phan_tich_pho(x, fs)
    cfg_auto, ly_do = cau_hinh_tu_dong(pho, fs, thai_nhi)

    # tín hiệu lọc nhẹ làm chuẩn so hình dạng sóng (giữ P, QRS, T)
    sos = ss.butter(2, [0.5, min(100, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    x_nhe = ss.sosfiltfilt(sos, x - np.median(x))

    ung_vien = [
        ("neurokit", None, "neurokit"),
        ("biosppy", None, "biosppy"),
        ("pantompkins1985", None, "pantompkins1985"),
        ("hamilton2002", None, "hamilton2002"),
        (pp_tuy_chinh, cfg_auto, "Tùy chỉnh – tự động cho file này"),
    ]
    bang = []
    for method, cfg, ten in ung_vien:
        try:
            clean, rp, rr, hr, q = xu_ly_ecg(x, fs, method, cfg)
        except Exception as e:                      # phương pháp không chạy được trên file này
            bang.append(dict(ten=ten, method=method, cfg=cfg, diem=0.0, loi=str(e)[:120]))
            continue
        d = dict(
            sqi=float(q) if np.isfinite(q) else 0.0,
            rr=_diem_rr(rp, fs, thai_nhi),
            hinh_dang=_diem_hinh_dang(clean, x_nhe, rp, fs, thai_nhi),
            khu_nhieu=_diem_khu_nhieu(clean, fs, thai_nhi),
        )
        tong = sum(TRONG_SO[k] * d[k] for k in TRONG_SO)
        f1 = doi_chieu_f1(rp, ref_doan, fs, dung_sai_ms) if ref_doan is not None and len(ref_doan) else None
        diem = f1 * 0.8 + tong * 0.2 if f1 is not None else tong      # có nhãn: F1 là tiêu chí chính
        bang.append(dict(ten=ten, method=method, cfg=cfg, diem=diem * 100, f1=f1,
                         nhip=60000 / np.mean(rr), so_nhip=len(rp), **d))
    bang.sort(key=lambda r: r["diem"], reverse=True)
    return dict(bang=bang, cfg_tu_dong=cfg_auto, ly_do=ly_do, pho=pho, doan_s=n / fs,
                co_nhan=ref_doan is not None and len(ref_doan) > 0)


def nhan_xet_phuong_phap(r: dict) -> str:
    """Giải thích ngắn vì sao một phương pháp được/không được đề xuất."""
    if "loi" in r:
        return "Không chạy được trên file này."
    yeu = []
    if r["hinh_dang"] < 0.8:
        yeu.append("làm méo hình dạng sóng (mất P/T)")
    if r["rr"] < 0.9:
        yeu.append("nhiều khoảng RR bất thường (bỏ sót hoặc bắt nhầm đỉnh R)")
    if r["sqi"] < 0.6:
        yeu.append("các nhịp sau lọc kém đồng nhất")
    if r["khu_nhieu"] < 0.7:
        yeu.append("còn nhiều nhiễu ngoài dải ECG")
    return ("Điểm yếu: " + "; ".join(yeu) + ".") if yeu else "Cân bằng tốt giữa khử nhiễu và giữ hình dạng sóng."
