"""
App phân tích tín hiệu ECG - Lab Tin học Vật lý Kỹ thuật
Chạy:  streamlit run app.py
"""
import html
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i18n import t
from xu_ly import (DS_DON_VI, DS_KIEU_NGUONG, DS_NEN, DS_NGUONG, DS_PHUONG_PHAP,
                   DS_UOC_LUONG, DS_WAVELET, DUOI_HO_TRO, GIOI_THIEU_PP, KHONG_CHON, MAC_DINH_LOC, PP_TUY_CHINH,
                   TEN_KIEU, TEN_NEN, TEN_NGUONG, TEN_PP, TEN_UOC, THIET_BI, cfg_sang_session, cfg_tu_session,
                   chi_so_don_vi_mac_dinh, doc_nhieu_file, doi_chieu, doi_don_vi, fmt, mo_ta_loc, nguong_nhip,
                   ten_phuong_an, tinh_hrv, tinh_khoang_song, xu_ly_ecg)
from bieu_do import (fig_dai_song, fig_dap_ung, fig_nhip_tim, fig_pho, fig_rr, fig_truoc_sau_loc, plotly_dai_song,
                     svg_sparkline, tao_pdf)
from de_xuat_loc import cau_hinh_tu_dong, de_xuat, nhan_xet_phuong_phap
from so_sanh_chi_so import (MAU, bien_thien_nhip_thai, danh_gia_chi_so, diem_luu_y, fig_khoang_tham_chieu, fig_shap,
                            khoang_str, so)

st.set_page_config(page_title="ECG", page_icon="🫀", layout="wide")
ss = st.session_state
ss.setdefault("lang", "vi")
ss.setdefault("chuyen_gia", False)
for k, v in MAC_DINH_LOC.items():
    ss.setdefault(k, v)
# Ghi lại giá trị các ô nhập ở đầu mỗi lần chạy: giữ giá trị khi ô đang ẩn (chế độ chuyên gia tắt) và khi đổi
# ngôn ngữ (nhãn đổi thì Streamlit coi là ô mới và sẽ trả về giá trị mặc định nếu không làm bước này).
GIU = list(MAC_DINH_LOC) + ["chuyen_gia", "nguon", "ten_bn", "tuoi", "gioi", "col_ecg", "thiet_bi", "fs_nhap",
                            "don_vi", "doi_tuong", "dung_sai", "t_start", "hien_s", "i_chon"]
for k in GIU:
    if k in ss:
        ss[k] = ss[k]

# ----------------------------------------------------------------------------
# Giao diện "phiếu lâm sàng"
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, input, textarea, button, td, th,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"], [data-baseweb="select"] div {
  font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1360px; }
[data-testid="stSidebar"] { background: #f2f6f8; border-right: 1px solid #dbe3e8; }
[data-testid="stSidebar"] h2 { font-size: 0.95rem; letter-spacing: .02em; color: #0b5c7a; margin-top: .6rem;
  padding-top: .6rem; border-top: 1px solid #dbe3e8; }
.ecg-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.ecg-brand .logo { width: 46px; height: 46px; border-radius: 12px; background: #0b5c7a; display: grid;
  place-items: center; flex: none; }
.ecg-brand h1 { font-size: 1.45rem; margin: 0; padding: 0; color: #1b2733; font-weight: 700; }
.ecg-brand .sub { color: #5b6b78; font-size: .84rem; }
.ecg-steps { display: flex; gap: 6px; margin: 14px 0 12px; }
.ecg-steps .st { flex: 1; display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 8px;
  background: #f2f6f8; color: #7a8a96; font-size: .85rem; font-weight: 500; border: 1px solid #e3e9ed; }
.ecg-steps .st b { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; flex: none;
  background: #dbe3e8; color: #5b6b78; font-size: .75rem; }
.ecg-steps .st.done { color: #1b2733; background: #eaf4f0; border-color: #cfe5da; }
.ecg-steps .st.done b { background: #2e7d32; color: #fff; }
.ecg-steps .st.now { color: #0b5c7a; background: #e6f1f6; border-color: #0b5c7a; font-weight: 600; }
.ecg-steps .st.now b { background: #0b5c7a; color: #fff; }
.ecg-patient { display: grid; grid-template-columns: 1.5fr .9fr 1fr 1.5fr 1fr 1.1fr .9fr; gap: 0;
  border: 1px solid #d5dde3; border-radius: 10px; overflow: hidden; margin-bottom: 14px; background: #fff; }
.ecg-patient div { padding: 8px 14px; border-right: 1px solid #edf1f4; }
.ecg-patient div:first-child { background: #0b5c7a; color: #fff; }
.ecg-patient small { display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em;
  color: #7a8a96; }
.ecg-patient div:first-child small { color: #bfe0ec; }
.ecg-patient span { font-weight: 600; font-size: .92rem; overflow-wrap: anywhere; }
@media (max-width: 1000px) { .ecg-patient { grid-template-columns: repeat(2, 1fr); } }
.ecg-cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 4px 0 16px; }
@media (max-width: 1000px) { .ecg-cards { grid-template-columns: repeat(2, 1fr); } }
.ecg-card { background: #fff; border: 1px solid #d5dde3; border-left: 5px solid var(--c); border-radius: 10px;
  padding: 10px 14px 8px; }
.ecg-card .lbl { color: #5b6b78; font-size: .78rem; font-weight: 500; }
.ecg-card .val { font-size: 1.7rem; font-weight: 700; color: #1b2733; line-height: 1.25; }
.ecg-card .val span { font-size: .8rem; font-weight: 500; color: #5b6b78; margin-left: 4px; }
.ecg-card .tag { font-size: .76rem; font-weight: 600; color: var(--c); }
.ecg-card .spark { color: var(--c); margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    c_l, c_e = st.columns([1, 1])
    c_l.radio("🌐", ["vi", "en"], key="lang", horizontal=True, label_visibility="collapsed",
              format_func=lambda v: {"vi": "Tiếng Việt", "en": "English"}[v])
    c_e.toggle(t("Chế độ chuyên gia"), key="chuyen_gia",
               help=t("Bật để chỉnh chi tiết bộ lọc, xem phổ tần số và các thông số kỹ thuật."))
chuyen_gia = ss["chuyen_gia"]

st.markdown(
    '<div class="ecg-brand"><div class="logo"><svg width="30" height="20" viewBox="0 0 30 20">'
    '<polyline points="1,11 8,11 10,15 13,2 16,18 19,11 29,11" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linejoin="round" stroke-linecap="round"/></svg></div><div>'
    f'<h1>{t("Phân tích điện tâm đồ")}</h1>'
    f'<div class="sub">{t("Lab Tin học Vật lý Kỹ thuật · Kết quả chỉ dùng cho học tập/nghiên cứu, không dùng để chẩn đoán.")}'
    '</div></div></div>', unsafe_allow_html=True)
khung_buoc = st.empty()
khung_bn = st.empty()


def ve_buoc(hien_tai: int):
    """Thanh tiến trình 4 bước; các bước trước 'hien_tai' đã xong."""
    ten = [t("Nhập dữ liệu"), t("Chọn bộ lọc"), t("Phân tích"), t("Xem & xuất báo cáo")]
    o = []
    for i, n in enumerate(ten):
        lop = "done" if i < hien_tai else "now" if i == hien_tai else ""
        o.append(f'<div class="st {lop}"><b>{"✓" if i < hien_tai else i + 1}</b>{n}</div>')
    khung_buoc.markdown('<div class="ecg-steps">' + "".join(o) + "</div>", unsafe_allow_html=True)


def the(nhan, gia_tri, don_vi="", tag="", mau="#0b5c7a", spark=""):
    return (f'<div class="ecg-card" style="--c:{mau}"><div class="lbl">{html.escape(nhan)}</div>'
            f'<div class="val">{gia_tri}<span>{html.escape(don_vi)}</span></div>'
            f'<div class="tag">{html.escape(tag)}</div><div class="spark">{spark}</div></div>')


MAU_TIN_CAY = {"Tốt": "#2e7d32", "Trung bình": "#ef8f00", "Kém": "#c62828"}

# ----------------------------------------------------------------------------
# Thanh bên: dữ liệu và bệnh nhân
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header(t("1. Dữ liệu"))
    nguon = st.radio(t("Nguồn dữ liệu"), ["Dùng dữ liệu mẫu", "Tải file lên"], format_func=t, key="nguon")
    files = []
    if nguon == "Tải file lên":
        # nhãn cố định (song ngữ) để file đã tải không bị mất khi đổi ngôn ngữ
        files = st.file_uploader(
            "File ECG · ECG files", type=DUOI_HO_TRO, accept_multiple_files=True, key="files",
            help="CSV/TXT · EDF/EDF+ (.edf) · Biopac (.acq) · PhysioNet WFDB (.hea + .dat; "
                 "+ .atr/.qrs/.ann)") or []

    st.header(t("2. Thông tin bệnh nhân"))
    if ss.get("ten_bn") in (None, "Bệnh nhân mẫu", "Sample patient"):
        ss["ten_bn"] = t("Bệnh nhân mẫu")
    ss.setdefault("tuoi", 25)
    ten = st.text_input(t("Họ tên / mã"), key="ten_bn")
    c_t, c_g = st.columns(2)
    tuoi = c_t.number_input(t("Tuổi"), 1, 120, key="tuoi")
    gioi = c_g.selectbox(t("Giới tính"), ["Nam", "Nữ"], format_func=t, key="gioi")

ve_buoc(0)

# ----------------------------------------------------------------------------
# Đọc dữ liệu
# ----------------------------------------------------------------------------
if nguon == "Dùng dữ liệu mẫu":
    fs_file = 500
    sig = nk.ecg_simulate(duration=60, sampling_rate=fs_file, heart_rate=75, noise=0.05, random_state=42)
    df = pd.DataFrame({"time_s": np.arange(len(sig)) / fs_file, "ecg_mV": sig})
    meta = dict(fs=None, units={}, ref=None, ky_hieu=None, mo_ta="", nguon_nhan="")
else:
    if not files:
        st.info(t("⬅️ Hãy tải file ECG lên ở thanh bên: CSV/TXT, EDF (.edf), Biopac (.acq), "
                  "hoặc dữ liệu PhysioNet WFDB (.hea + .dat, kèm nhãn .atr/.qrs nếu có)."))
        st.stop()
    try:
        df, meta = doc_nhieu_file(files)
    except Exception as e:
        st.error(t("Không đọc được file: {e}", e=e))
        st.stop()
    fs_file = int(round(meta["fs"])) if meta["fs"] else 500
ten_file = ",".join(sorted(f.name for f in files)) if nguon == "Tải file lên" else "du_lieu_mau"

# ----------------------------------------------------------------------------
# Thanh bên: tín hiệu
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header(t("3. Tín hiệu"))
    cols_num = df.select_dtypes(include="number").columns.tolist()
    if not cols_num:
        st.error(t("File không có cột số."))
        st.stop()
    la_thoi_gian = lambda c: "time" in c.lower() or c.lower() in ("t", "s")
    goi_y = ([c for c in cols_num if any(k in c.lower() for k in ("direct", "ecg", "ekg", "mlii"))]
             or [c for c in cols_num if not la_thoi_gian(c)] or cols_num)
    if ss.get("col_ecg") not in cols_num:
        ss["col_ecg"] = goi_y[0]
    col_ecg = st.selectbox(t("Cột tín hiệu ECG"), cols_num, key="col_ecg")

    col_time = next((c for c in cols_num if la_thoi_gian(c)), None)
    if meta["fs"]:
        col_time = "fs"
    elif col_time is not None and len(df) > 2:
        dt = np.median(np.diff(df[col_time].values[:1000]))
        if dt > 0:
            fs_file = int(round(1 / dt))
    fs_chua_ro = col_time is None                     # file không ghi tần số lấy mẫu
    co_don_vi = bool(meta["units"].get(col_ecg))
    v = pd.to_numeric(df[col_ecg], errors="coerce").dropna()
    dv_mac_dinh = DS_DON_VI[chi_so_don_vi_mac_dinh(meta["units"].get(col_ecg, ""), v)]

    def ap_thiet_bi():
        p = THIET_BI.get(ss.get("thiet_bi"))
        if p:
            if ss.get("_fs_chua_ro", True):
                ss["fs_nhap"] = p[0]
            if not ss.get("_co_don_vi", False):
                ss["don_vi"] = p[1]

    ss["_fs_chua_ro"], ss["_co_don_vi"] = fs_chua_ro, co_don_vi
    goi_y_thai = any(k in " ".join(cols_num + [f.name for f in files]).lower()
                     for k in ("direct", "abdomen", "fetal", "fecg", "adfecg"))
    if ss.get("_nguon") != (ten_file, col_ecg, fs_file):        # đổi file/kênh → về giá trị của file
        if ss.get("_nguon", ("",))[0] != ten_file:
            ss["doi_tuong"] = "Thai nhi (FECG)" if goi_y_thai else "Người lớn"
        ss["_nguon"] = (ten_file, col_ecg, fs_file)
        ss["fs_nhap"], ss["don_vi"] = fs_file, dv_mac_dinh
        ap_thiet_bi()

    thiet_bi = st.selectbox(t("Thiết bị đo"), list(THIET_BI), key="thiet_bi", format_func=t, on_change=ap_thiet_bi,
                            help=t("Chọn máy đo để app tự điền tần số lấy mẫu và đơn vị. "
                                   "Thông tin ghi trong file luôn được ưu tiên."))
    if THIET_BI.get(thiet_bi):
        st.caption("ℹ️ " + t(THIET_BI[thiet_bi][2]) + ("" if fs_chua_ro else " " + t("Tần số ghi trong file được ưu tiên.")))
    fs = int(st.number_input(t("Tần số lấy mẫu (Hz)"), 50, 10000, key="fs_nhap"))
    if meta["fs"]:
        st.caption(t("Đọc từ file {mt}: {fs} Hz, {n} kênh.", mt=meta["mo_ta"], fs=f"{meta['fs']:g}", n=len(cols_num))
                   + (" " + t("Bỏ {k} kênh có tần số khác.", k=meta["bo_kenh"]) if meta.get("bo_kenh") else ""))
    if fs_chua_ro and not THIET_BI.get(thiet_bi):
        st.warning(t("File không có cột thời gian: hãy chọn **thiết bị đo** hoặc nhập **đúng tần số lấy mẫu**, "
                     "nếu sai thì nhịp tim và mọi khoảng thời gian sẽ sai theo."))
    don_vi = st.selectbox(t("Đơn vị tín hiệu"), DS_DON_VI, key="don_vi", format_func=t,
                          help=t("Đơn vị ghi trong file: {u}", u=meta["units"][col_ecg]) if co_don_vi else None)

    doi_tuong = st.selectbox(t("Đối tượng"), ["Người lớn", "Thai nhi (FECG)"], key="doi_tuong", format_func=t, help=t("Đổi ngưỡng đánh giá: người lớn 60–100 l/p, thai nhi 110–160 l/p (FIGO)."))
    thai_nhi = doi_tuong.startswith("Thai")

    dung_sai = None
    if meta["ref"] is not None:
        st.markdown(t("**Nhãn chuẩn:** {n} đỉnh R ({src})", n=len(meta["ref"]), src=meta["nguon_nhan"]))
        mac_dinh_ds = 50 if thai_nhi else 150
        if ss.get("_ds_cho") != thai_nhi:                 # đổi đối tượng → về dung sai chuẩn
            ss["_ds_cho"], ss["dung_sai"] = thai_nhi, mac_dinh_ds
        dung_sai = (st.slider(t("Dung sai khớp đỉnh R (ms)"), 10, 150, step=10, key="dung_sai",
                              help=t("Chuẩn ANSI/AAMI EC57 dùng 150 ms cho người lớn; ECG thai nhi nhịp nhanh "
                                     "nên thường dùng 50 ms.")) if chuyen_gia else mac_dinh_ds)

# ----------------------------------------------------------------------------
# Thanh bên: bộ lọc
# ----------------------------------------------------------------------------
def ap_dung_loc(method: str, cfg: dict | None):
    """Ghi phương án được chọn vào các ô điều khiển rồi tự phân tích lại."""
    ss["method"] = method
    if cfg:
        ss.update(cfg_sang_session(cfg))
    ss["tu_chay"] = True


with st.sidebar:
    st.header(t("4. Bộ lọc"))
    hi_max = float(np.floor(fs / 2 * 0.99 * 10) / 10)
    ss["cf_hi"] = min(float(ss["cf_hi"]), hi_max)
    st.button(t("🧭 Đề xuất bộ lọc cho file này"), use_container_width=True,
              on_click=lambda: ss.update(can_de_xuat=True),
              help=t("App tự thử mọi cách lọc trên file này và gợi ý cách tốt nhất (mất vài giây)."))
    method = st.selectbox(t("Phương pháp lọc"), DS_PHUONG_PHAP, key="method", format_func=lambda m: t(TEN_PP[m]))
    phu_hop, luu_y_pp, thuat_toan = GIOI_THIEU_PP[method]
    st.caption(f"✅ {t(phu_hop)}  \n⚠️ {t(luu_y_pp)}" + (f"  \n{t('Thuật toán')}: {thuat_toan}" if chuyen_gia else ""))
    cfg = None
    if method == PP_TUY_CHINH and not chuyen_gia:
        cfg = cfg_tu_session(ss)
        st.caption("⚙️ " + mo_ta_loc(method, cfg) + ". " + t("Bật **Chế độ chuyên gia** để chỉnh từng bước lọc."))
    elif method == PP_TUY_CHINH:
        st.markdown(t("**① Giữ dải tần của tim** · Butterworth"))
        st.slider(t("Độ dứt khoát khi cắt (bậc)"), 1, 8, key="cf_bac",
                  help=t("Cao hơn: cắt gọn hơn nhưng dễ tạo gợn quanh QRS. Bậc 2–4 là đủ cho ECG."))
        c_lo, c_hi = st.columns(2)
        lo = c_lo.number_input(t("Bỏ phần dưới (Hz)"), 0.05, 5.0, step=0.05, format="%.2f", key="cf_lo",
                               help=t("0,5 Hz: làm phẳng đường nền. 0,05 Hz: giữ nguyên đoạn ST."))
        hi = c_hi.number_input(t("Bỏ phần trên (Hz)"), 5.0, hi_max, step=1.0, format="%.1f", key="cf_hi",
                               help=t("40 Hz: chuẩn khi theo dõi, loại nhiễu cơ. Tối đa {m} Hz với file này.", m=f"{hi_max:g}"))
        if lo >= hi:
            st.error(t("Số ở ô \"Bỏ phần dưới\" phải nhỏ hơn ô \"Bỏ phần trên\"."))
            st.stop()
        st.radio(t("Làm phẳng đường nền bằng"), DS_NEN, key="cf_nen", format_func=lambda v: t(TEN_NEN[v]),
                 help=t("Đường nền trôi lên xuống do thở hoặc cử động. Wavelet bám nền tốt hơn khi trôi mạnh."))

        st.markdown(t("**② Lọc nhiễu điện lưới** · Notch"))
        notch = st.checkbox(t("Bật lọc nhiễu điện"), key="cf_notch",
                            help=t("Loại nhiễu từ ổ điện, đèn, máy móc xung quanh."))
        c_f, c_q = st.columns(2)
        c_f.selectbox(t("Tần số điện"), [50, 60], disabled=not notch, key="cf_fnotch", help=t("Việt Nam dùng điện 50 Hz."))
        c_q.number_input(t("Độ hẹp (Q)"), 1.0, 100.0, step=1.0, disabled=not notch, key="cf_Q",
                         help=t("Lớn hơn: chỉ cắt đúng tần số điện, ít ảnh hưởng tín hiệu tim."))
        st.checkbox(t("Lọc cả bội số (100, 150 Hz)"), disabled=not notch, key="cf_hai")

        st.markdown(t("**③ Làm mịn tín hiệu** · Wavelet"))
        wl_on = st.checkbox(t("Bật làm mịn"), key="cf_wl", help=t("Giảm nhiễu cơ và nhiễu lăn tăn trên đường sóng."))
        st.selectbox(t("Dạng sóng mẫu"), DS_WAVELET, key="cf_wavelet", disabled=not wl_on,
                     help=t("sym8 và db4 có hình giống phức bộ QRS nên hợp với ECG."))
        muc = st.slider(t("Mức làm mịn"), 1, 8, disabled=not wl_on, key="cf_muc",
                        help=t("Cao hơn: mịn hơn nhưng dễ mất sóng P, T. Mức 3–5 là hợp lý."))
        if wl_on and fs / 2 ** (muc + 1) < 5:
            st.warning(t("Mức làm mịn này có thể làm mất sóng P và T. Nên giảm xuống."))
        st.selectbox(t("Độ mạnh"), DS_NGUONG, disabled=not wl_on, key="cf_nguong", format_func=lambda v: t(TEN_NGUONG[v]))
        c_m, c_t2 = st.columns(2)
        c_m.radio(t("Kiểu làm mịn"), DS_KIEU_NGUONG, disabled=not wl_on, key="cf_mem", format_func=lambda v: t(TEN_KIEU[v]))
        c_t2.radio(t("Ước lượng nhiễu"), DS_UOC_LUONG, disabled=not wl_on, key="cf_theomuc",
                   format_func=lambda v: t(TEN_UOC[v]))
        cfg = cfg_tu_session(ss)
    chay = st.button(t("▶ Phân tích"), type="primary", use_container_width=True) or ss.pop("tu_chay", False)

ecg_raw = doi_don_vi(pd.to_numeric(df[col_ecg], errors="coerce").interpolate().bfill().values, don_vi)
duration = len(ecg_raw) / fs
khoa_dau_vao = (ten_file, len(ecg_raw), col_ecg, fs, don_vi, method, tuple(sorted((cfg or {}).items())))
ref = meta["ref"] if meta["ref"] is not None and (not meta["fs"] or fs == int(round(meta["fs"]))) else None
if ss.get("khoa") != khoa_dau_vao:
    ss.pop("kq", None)

# ----------------------------------------------------------------------------
# Thanh thông tin bệnh nhân
# ----------------------------------------------------------------------------
o_bn = [(t("Bệnh nhân"), ten), (t("Tuổi / Giới"), f"{tuoi} · {t(gioi)}"), (t("Đối tượng"), t(doi_tuong)),
        (t("Dữ liệu"), (meta["mo_ta"] or ten_file) if nguon == "Tải file lên" else t("Dữ liệu mẫu (mô phỏng)")),
        (t("Kênh"), col_ecg), (t("Ghi"), f"{duration:.1f} s · {fs} Hz"),
        (t("Ngày"), datetime.now().strftime("%d/%m/%Y"))]
khung_bn.markdown('<div class="ecg-patient">' + "".join(
    f"<div><small>{html.escape(k)}</small><span>{html.escape(str(v))}</span></div>" for k, v in o_bn) + "</div>",
    unsafe_allow_html=True)
ve_buoc(1)

if chuyen_gia:
    with st.expander(t("Xem 10 dòng đầu của dữ liệu")):
        st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# Đề xuất bộ lọc cho file này
# ----------------------------------------------------------------------------
khoa_de_xuat = (ten_file, len(ecg_raw), col_ecg, fs, don_vi, doi_tuong, dung_sai)
if ss.pop("can_de_xuat", False):
    with st.spinner(t("Đang phân tích phổ và chạy thử các phương pháp lọc...")):
        ss.de_xuat = dict(khoa=khoa_de_xuat, kq=de_xuat(ecg_raw, fs, thai_nhi, xu_ly_ecg, PP_TUY_CHINH,
                                                        ref=ref, dung_sai_ms=dung_sai or 50))
dx = ss.get("de_xuat")
if dx and dx["khoa"] == khoa_de_xuat:
    kqdx = dx["kq"]
    tot = next(r for r in kqdx["bang"] if "loi" not in r)
    with st.expander(t("🧭 Đề xuất bộ lọc cho file này"), expanded=True):
        st.success(t("**Nên dùng: {ten}** · {d}/100 điểm", ten=ten_phuong_an(tot), d=f"{tot['diem']:.0f}"))
        tieu_chi = (t("chủ yếu dựa trên mức khớp với nhãn đỉnh R của bác sĩ đi kèm file") if kqdx["co_nhan"] else
                    t("nhịp rõ và đều, bắt nhịp đúng, giữ hình dạng sóng, sạch nhiễu"))
        st.caption(t("App đã thử 5 cách lọc trên {s} giây đầu của bản ghi. Điểm {tc}.",
                     s=f"{kqdx['doan_s']:.0f}", tc=tieu_chi))
        hang = []
        for i, r in enumerate(kqdx["bang"], 1):
            d = {t("Hạng"): i, t("Cách lọc"): ten_phuong_an(r), t("Điểm"): 0 if "loi" in r else round(r["diem"]),
                 t("Nhận xét"): nhan_xet_phuong_phap(r)}
            if "loi" not in r and chuyen_gia:
                d.update({t("Nhịp rõ, đều"): f"{r['sqi'] * 100:.0f}%", t("Bắt nhịp đúng"): f"{r['rr'] * 100:.0f}%",
                          t("Giữ hình dạng sóng"): f"{r['hinh_dang'] * 100:.0f}%",
                          t("Sạch nhiễu"): f"{r['khu_nhieu'] * 100:.0f}%"})
            if "loi" not in r and kqdx["co_nhan"]:
                d[t("Khớp nhãn bác sĩ")] = f"{r['f1'] * 100:.1f}%"
            hang.append(d)
        st.dataframe(pd.DataFrame(hang), use_container_width=True, hide_index=True)
        st.markdown(t("**App nhận thấy gì về nhiễu của file này:**"))
        for l in cau_hinh_tu_dong(kqdx["pho"], fs, thai_nhi)[1]:
            st.markdown(f"- {l}")
        ds_ok = [r for r in kqdx["bang"] if "loi" not in r]
        c_a, c_b = st.columns([3, 2])
        if ss.get("i_chon", 0) >= len(ds_ok):
            ss["i_chon"] = 0
        i_chon = c_a.selectbox(t("Phương án áp dụng"), range(len(ds_ok)), format_func=lambda i: ten_phuong_an(ds_ok[i]),
                               label_visibility="collapsed", key="i_chon")
        c_b.button(t("✅ Áp dụng và phân tích"), type="primary", use_container_width=True,
                   on_click=ap_dung_loc, args=(ds_ok[i_chon]["method"], ds_ok[i_chon]["cfg"]))

if not chay and "kq" not in ss:
    st.info(t("Bấm **▶ Phân tích** ở thanh bên để bắt đầu."))
    st.stop()

# ----------------------------------------------------------------------------
# Phân tích
# ----------------------------------------------------------------------------
if chay or "kq" not in ss:
    with st.spinner(t("Đang lọc tín hiệu, tìm đỉnh R và tính chỉ số...")):
        try:
            ecg_clean, rpeaks, rr_ms, hr, quality = xu_ly_ecg(ecg_raw, fs, method, cfg)
        except Exception as e:
            st.error(t("Lỗi khi phân tích: {e}", e=e))
            st.stop()
        khoang = tinh_khoang_song(ecg_clean, rpeaks, fs)
        hrv = tinh_hrv(rpeaks, fs) if duration >= 30 else {}
    ss.kq = dict(ecg_clean=ecg_clean, rpeaks=rpeaks, rr_ms=rr_ms, hr=hr, quality=quality, khoang=khoang, hrv=hrv)
    ss.khoa = khoa_dau_vao
ve_buoc(3)

kq = ss.kq
ecg_clean, rpeaks, rr_ms, hr = kq["ecg_clean"], kq["rpeaks"], kq["rr_ms"], kq["hr"]
khoang, hrv, quality = kq["khoang"], kq["hrv"], kq["quality"]
hr_mean = 60000 / np.mean(rr_ms)
dc = doi_chieu(rpeaks, ref, fs, dung_sai) if ref is not None and len(ref) else None

ty_le_lech = float(np.mean(np.abs(np.diff(rr_ms)) / rr_ms[:-1] > 0.2)) if len(rr_ms) > 2 else 0.0
if thai_nhi:
    gia_tri = {"Nhịp tim thai cơ bản": float(np.median(hr)), "Biến thiên nhịp tim thai": bien_thien_nhip_thai(rpeaks, fs)}
else:
    gia_tri = {"Nhịp tim": hr_mean, "PR": khoang.get("PR (ms)", np.nan), "QRS": khoang.get("QRS (ms)", np.nan),
               "QTc (Bazett)": khoang.get("QTc Bazett (ms)", np.nan),
               "SDNN": hrv.get("SDNN (ms)", np.nan), "RMSSD": hrv.get("RMSSD (ms)", np.nan)}
dong_ss = danh_gia_chi_so(gia_tri, thai_nhi, gioi, quality, duration, ty_le_lech)
theo_ten = {r["ten"]: r for r in dong_ss}
luu_y = diem_luu_y(dong_ss)
n_ngoai = sum(r["trang_thai"] in ("Cao", "Thấp") for r in dong_ss)
n_danh_gia = sum(r["trang_thai"] != "Không đánh giá" for r in dong_ss)
muc_sqi = ("Tốt" if quality >= 0.85 else "Trung bình" if quality >= 0.7 else "Kém") if np.isfinite(quality) else None

# ----------------------------------------------------------------------------
# Cảnh báo tần số lấy mẫu
# ----------------------------------------------------------------------------
ngoai_sinh_ly = not ((50 <= hr_mean <= 240) if thai_nhi else (35 <= hr_mean <= 200))
dang_ngo = fs_chua_ro and not ((100 <= hr_mean <= 180) if thai_nhi else (45 <= hr_mean <= 120))
if ngoai_sinh_ly or dang_ngo:
    st.warning(
        t("**Nhịp tim {hr} l/p cao bất thường, hãy kiểm tra tần số lấy mẫu.** " if hr_mean > 100 else
          "**Nhịp tim {hr} l/p thấp bất thường, hãy kiểm tra tần số lấy mẫu.** ", hr=f"{hr_mean:.0f}")
        + (t("File không ghi tần số lấy mẫu nên app đang dùng **{fs} Hz** (theo thiết bị/ô nhập). "
             "Nếu con số này khác tần số thật của máy đo, nhịp tim và mọi khoảng thời gian đều sai theo cùng tỷ lệ.",
             fs=fs) if fs_chua_ro else t("Tần số lấy mẫu đang dùng có thể chưa đúng.")))
    if fs_chua_ro:
        cac_fs = [f for f in (125, 200, 250, 256, 360, 500, 1000) if f != fs]
        st.caption(t("Ước tính nhịp tim nếu tần số lấy mẫu thật là: ")
                   + " · ".join(f"{f} Hz ≈ {hr_mean * f / fs:.0f} " + t("l/p") for f in cac_fs)
                   + t(". Hãy xem cấu hình máy đo (ví dụ chu kỳ đọc ADC trong code Arduino) để biết con số đúng."))
if don_vi.startswith("ADC 1"):
    st.caption(t("Biên độ mV được quy đổi từ giá trị ADC theo giả định AD8232 (khuếch đại 1100), chỉ mang tính ước lượng."))

# ----------------------------------------------------------------------------
# Thẻ chỉ số chính
# ----------------------------------------------------------------------------
r_hr = theo_ten.get("Nhịp tim thai cơ bản" if thai_nhi else "Nhịp tim")
band = nguong_nhip(thai_nhi)
mau_hr = MAU.get(r_hr["trang_thai"], "#9e9e9e")
the_html = [
    the(t("Nhịp tim thai cơ bản") if thai_nhi else t("Nhịp tim trung bình"),
        f"{r_hr['x']:.0f}", t("l/p"), f"{t(r_hr['trang_thai'])} · {band[0]}–{band[1]}", mau_hr,
        svg_sparkline(hr, band)),
    the(t("Thấp nhất / cao nhất"), f"{hr.min():.0f}–{hr.max():.0f}", t("l/p"), t("nhịp tức thời"), "#0b5c7a"),
    the(t("Số nhịp"), f"{len(rpeaks)}", "", t("trong {s} s ghi", s=f"{duration:.0f}"), "#0b5c7a"),
    the(t("Chất lượng tín hiệu"), "—" if muc_sqi is None else f"{quality * 100:.0f}", "%",
        t(muc_sqi) if muc_sqi else t("không đánh giá được"), MAU_TIN_CAY.get(muc_sqi, "#9e9e9e")),
    the(t("Ngoài khoảng tham chiếu"), f"{n_ngoai}", f"/ {n_danh_gia}",
        t("xem tab So sánh chỉ số") if n_ngoai else t("tất cả trong khoảng"),
        "#c62828" if n_ngoai else "#2e7d32"),
]
st.markdown('<div class="ecg-cards">' + "".join(the_html) + "</div>", unsafe_allow_html=True)

if dc is not None:
    st.subheader(t("🎯 Đối chiếu với nhãn chuẩn"))
    n1, n2, n3, n4 = st.columns(4)
    n1.metric(t("Độ nhạy Se"), f"{dc['Se']*100:.1f}%", help=t("Tỷ lệ nhãn chuẩn được app tìm thấy = TP/(TP+FN)"))
    n2.metric(t("Giá trị dự báo dương PPV"), f"{dc['PPV']*100:.1f}%", help=t("Tỷ lệ đỉnh app tìm là đúng = TP/(TP+FP)"))
    n3.metric("F1", f"{dc['F1']*100:.1f}%")
    n4.metric(t("Lệch thời gian TB"), f"{dc['lech_ms']:.1f} ms")
    st.caption(t("TP = {tp} · bỏ sót FN = {fn} · phát hiện thừa FP = {fp} · dung sai ±{ds} ms · nhãn: {src}.",
                 tp=dc["TP"], fn=dc["FN"], fp=dc["FP"], ds=dung_sai, src=meta["nguon_nhan"]))
    if thai_nhi and "abdomen" in col_ecg.lower():
        st.info(t("Kênh bụng mẹ (Abdomen) chứa cả ECG của mẹ (biên độ lớn) lẫn của thai nhi, nên thuật toán chủ yếu "
                  "bắt nhịp **mẹ**. Nhãn chuẩn là nhịp **thai**, vì vậy Se/PPV thấp là đúng bản chất tín hiệu. "
                  "Hãy chọn kênh **Direct_1** (điện cực đặt trên đầu thai nhi) để đánh giá thuật toán."))

tab1, tab2, tab3, tab4, tab5 = st.tabs([t("📈 Dải sóng ECG"), t("🧹 Trước/sau lọc"), t("❤️ Nhịp tim & HRV"),
                                        t("🩺 So sánh chỉ số"), t("📄 Báo cáo")])

with tab1:
    c_s, c_d = st.columns([4, 1])
    max_start = max(0.0, duration - 10)
    ss["t_start"] = min(float(ss.get("t_start", 0.0)), float(max_start))
    ss.setdefault("hien_s", 10)
    t_start = (c_s.slider(t("Bắt đầu từ (s)"), 0.0, float(max_start), step=0.5, key="t_start")
               if max_start > 0 else 0.0)
    hien_s = c_d.select_slider(t("Khung nhìn (s)"), [5, 10, 20, 30], key="hien_s")
    try:
        t0 = 0.0 if duration <= 120 else max(0.0, t_start - 10)
        t1 = duration if duration <= 120 else min(duration, t_start + 60)
        fig_tt = plotly_dai_song(ecg_clean, rpeaks, fs, t0, t1, ref, hien_s)
        fig_tt.update_xaxes(range=[t_start, min(t_start + hien_s, t1)])
        st.plotly_chart(fig_tt, use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
        st.caption(t("Kéo để dời theo thời gian · cuộn chuột hoặc kéo trên trục để phóng to · nhấp đúp để về khung ban đầu · "
                     "rê chuột lên chấm xanh để xem nhịp tim tức thời.")
                   + (" " + t("Tam giác đỏ: nhãn chuẩn.") if ref is not None else ""))
    except ImportError:
        st.info(t("Chưa cài plotly nên hiển thị dải sóng tĩnh."))
    fig_strip = fig_dai_song(ecg_clean, rpeaks, fs, t_start, min(10, duration), ref)
    with st.expander(t("Dải sóng chuẩn 25 mm/s, 10 mm/mV (bản in trong báo cáo)")):
        st.pyplot(fig_strip)

with tab2:
    fig_loc = fig_truoc_sau_loc(ecg_raw, ecg_clean, fs)
    st.pyplot(fig_loc)
    st.caption(t("Cách lọc đang dùng: {m}", m=mo_ta_loc(method, cfg)))
    fig_pho_, fig_dap_ung_ = fig_pho(ecg_raw, ecg_clean, fs), None
    if chuyen_gia:
        st.pyplot(fig_pho_)
        if method == PP_TUY_CHINH:
            fig_dap_ung_ = fig_dap_ung(fs, cfg)
            st.pyplot(fig_dap_ung_)
            st.caption(t("Đường cong cho biết dải tần được giữ lại (gần 0 dB) và dải bị loại bỏ. Phần làm mịn wavelet "
                         "không thể hiện ở đây; xem hiệu quả của nó ở biểu đồ phổ phía trên."))
    else:
        st.caption(t("Bật Chế độ chuyên gia để xem phổ tần số trước/sau lọc."))

with tab3:
    fig_hr = fig_nhip_tim(rpeaks, hr, fs, band)
    st.pyplot(fig_hr)
    fig_rr_ = fig_rr(rr_ms)
    st.pyplot(fig_rr_)
    if not hrv:
        st.warning(t("Tín hiệu ngắn hơn 30 s nên không tính HRV. Nên ghi tối thiểu 5 phút cho HRV ngắn hạn."))

# Bảng số liệu đo chi tiết
rows = [(t("Nhịp tim trung bình (l/p)"), fmt(hr_mean, 0)),
        (t("Nhịp tim min / max (l/p)"), f"{hr.min():.0f} / {hr.max():.0f}"),
        (t("RR trung bình (ms)"), fmt(np.mean(rr_ms), 0))]
rows += [(k, fmt(v, 0)) for k, v in khoang.items()]
rows += [(k, fmt(v, 1)) for k, v in hrv.items()]
if dc is not None:
    rows += [(t("Đối chiếu nhãn chuẩn") + ": Se (%)", fmt(dc["Se"] * 100, 1)),
             (t("Đối chiếu nhãn chuẩn") + ": PPV (%)", fmt(dc["PPV"] * 100, 1)),
             (t("Đối chiếu nhãn chuẩn") + ": F1 (%)", fmt(dc["F1"] * 100, 1)),
             (t("Đối chiếu nhãn chuẩn") + ": TP / FN / FP", f"{dc['TP']} / {dc['FN']} / {dc['FP']}")]
bang = pd.DataFrame(rows, columns=[t("Chỉ số"), t("Giá trị")])
C_DG = t("Đánh giá")
bang_ss = pd.DataFrame([{
    t("Chỉ số"): t(r["ten"]), t("Giá trị"): "—" if not np.isfinite(r["x"]) else f"{so(r['x'])} {t(r['dv'])}",
    t("Khoảng tham chiếu"): khoang_str(r), C_DG: t(r["trang_thai"]), t("Độ tin cậy đo"): t(r["tin_cay"]),
    t("Đóng góp SHAP"): round(r["phi"], 2), t("Nguồn tham chiếu"): t(r["nguon"])} for r in dong_ss])

with tab4:
    st.info(t("Phần này **không đưa ra chẩn đoán**. Mỗi chỉ số được đặt cạnh khoảng tham chiếu đã công bố "
              "để bác sĩ đánh giá nhanh; kết luận lâm sàng thuộc về bác sĩ."), icon="🩺")
    k1, k2, k3 = st.columns(3)
    k1.metric(t("Chỉ số trong khoảng tham chiếu"), f"{n_danh_gia - n_ngoai}/{n_danh_gia}")
    k2.metric(t("Chỉ số ngoài khoảng"), n_ngoai)
    k3.metric(t("Điểm lệch tổng hợp"), f"{sum(r['phi'] for r in dong_ss):.2f}",
              help=t("0 = mọi chỉ số nằm trong khoảng tham chiếu. Không phải điểm nguy cơ."))
    fig_ref = fig_khoang_tham_chieu(dong_ss)
    st.pyplot(fig_ref)
    st.markdown(t("**Chỉ số nào đóng góp vào mức lệch?** (biểu đồ thác nước kiểu SHAP)"))
    fig_shap_ = fig_shap(dong_ss)
    st.pyplot(fig_shap_)
    with st.expander(t("Cách đọc biểu đồ SHAP và độ tin cậy")):
        st.markdown(t(
            "- Biểu đồ bắt đầu từ **giá trị nền 0**, nghĩa là mọi chỉ số nằm trong khoảng tham chiếu. Mỗi thanh "
            "là phần đóng góp của một chỉ số; cộng dồn lại ra **điểm lệch tổng hợp**.\n"
            "- Đóng góp = mức lệch chuẩn hoá ra ngoài khoảng (1 = lệch thêm một đoạn bằng độ rộng khoảng, tối đa 2) "
            "× hệ số độ tin cậy của phép đo (Tốt 1,0 · Trung bình 0,7 · Kém 0,4). Các chỉ số có trọng số bằng nhau.\n"
            "- Vì điểm lệch là tổng của từng chỉ số (mô hình cộng tính), giá trị SHAP của mỗi chỉ số **đúng bằng** "
            "phần đóng góp của nó, không phải ước lượng như khi giải thích mô hình học máy hộp đen.\n"
            "- **Độ tin cậy đo** dựa trên chất lượng tín hiệu (SQI). Các khoảng PR/QRS/QTc đo từ hình thái sóng nên "
            "được hạ một bậc. HRV cần bản ghi ≥ 5 phút và ít nhịp lệch.\n"
            "- Đỏ = cao hơn khoảng tham chiếu, xanh dương = thấp hơn, xám = trong khoảng."))

    st.markdown(t("**Bảng so sánh**"))
    mau_dg = {t("Cao"): "color:#c62828;font-weight:600", t("Thấp"): "color:#1565c0;font-weight:600",
              t("Bình thường"): "color:#2e7d32"}
    try:
        sty = bang_ss.style.map(lambda v: mau_dg.get(v, ""), subset=[C_DG])
    except AttributeError:
        sty = bang_ss.style.applymap(lambda v: mau_dg.get(v, ""), subset=[C_DG])
    st.dataframe(sty.format({t("Đóng góp SHAP"): "{:.2f}"}), use_container_width=True, hide_index=True)

    st.markdown(t("**Các điểm cần bác sĩ lưu ý**"))
    for l in luu_y or [t("Tất cả chỉ số đánh giá được đều nằm trong khoảng tham chiếu.")]:
        st.markdown(f"- {l}")
    if thai_nhi:
        st.caption(t("Bản ghi thai nhi: chỉ so sánh nhịp tim thai cơ bản và biến thiên theo FIGO; "
                     "không đánh giá PR/QRS/QT theo ngưỡng người lớn."))
    elif tuoi < 18:
        st.caption(t("Khoảng tham chiếu đang dùng là của người lớn; trẻ em có ngưỡng khác theo lứa tuổi."))

    df_beats = pd.DataFrame({t("Nhịp số"): np.arange(1, len(rr_ms) + 1),
                             t("Thời điểm R (s)"): np.round(rpeaks[1:] / fs, 3),
                             "RR (ms)": np.round(rr_ms, 1), t("Nhịp tim (l/p)"): np.round(hr, 1)})
    with st.expander(t("Số liệu đo chi tiết")):
        st.dataframe(bang, use_container_width=True, hide_index=True)
        st.dataframe(df_beats, use_container_width=True, hide_index=True, height=250)
    d1, d2, d3 = st.columns(3)
    d1.download_button(t("⬇ Bảng so sánh (CSV)"), bang_ss.to_csv(index=False).encode("utf-8-sig"),
                       "so_sanh_chi_so.csv", "text/csv", use_container_width=True)
    d2.download_button(t("⬇ Số liệu đo (CSV)"), bang.to_csv(index=False).encode("utf-8-sig"),
                       "chi_so_ecg.csv", "text/csv", use_container_width=True)
    d3.download_button(t("⬇ Từng nhịp (CSV)"), df_beats.to_csv(index=False).encode("utf-8-sig"),
                       "tung_nhip_ecg.csv", "text/csv", use_container_width=True)

with tab5:
    bay_gio = datetime.now()
    thong_tin = {
        t("Họ tên / mã"): ten, t("Tuổi"): tuoi, t("Giới tính"): t(gioi), t("Đối tượng"): t(doi_tuong),
        t("Dữ liệu"): ten_file if nguon == "Tải file lên" else t("Dữ liệu mẫu (mô phỏng)"), t("Kênh"): col_ecg,
        t("Thiết bị đo"): t(thiet_bi) if THIET_BI.get(thiet_bi) else (meta["mo_ta"] or "—"),
        t("Tần số lấy mẫu"): f"{fs} Hz", t("Thời lượng ghi"): f"{duration:.1f} s",
        t("Phương pháp lọc"): mo_ta_loc(method, cfg),
        "_ma": "ECG-" + bay_gio.strftime("%y%m%d-%H%M"), "_ngay": bay_gio.strftime("%d/%m/%Y %H:%M"),
    }
    if thai_nhi:
        r_bt = theo_ten["Biến thiên nhịp tim thai"]
        kpi = [(t("Nhịp tim thai cơ bản"), fmt(r_hr["x"], 0), t("l/p"), r_hr["trang_thai"]),
               (t("Biến thiên nhịp tim thai"), "—" if not np.isfinite(r_bt["x"]) else so(r_bt["x"]), t("l/p"),
                r_bt["trang_thai"]),
               (t("Số nhịp"), str(len(rpeaks)), "", None),
               (t("Chất lượng tín hiệu"), "—" if muc_sqi is None else f"{quality * 100:.0f}", "%", None)]
    else:
        kpi = [(t("Nhịp tim TB"), fmt(hr_mean, 0), t("l/p"), r_hr["trang_thai"])]
        for k in ("PR", "QRS", "QTc (Bazett)"):
            r = theo_ten[k]
            kpi.append((t(k), fmt(r["x"], 0), "ms", r["trang_thai"]))
        kpi.append((t("Chất lượng tín hiệu"), "—" if muc_sqi is None else f"{quality * 100:.0f}", "%", None))
    st.markdown(t("Trang đầu là **phiếu kết quả** khổ A4: thông tin bệnh nhân, ô chỉ số chính, dải sóng 10 giây, bảng so "
                  "sánh với khoảng tham chiếu, các điểm cần lưu ý và ô để bác sĩ ghi kết luận, ký tên. Các trang sau là "
                  "biểu đồ chi tiết."))
    figs_pdf = ([fig_ref, fig_shap_, fig_strip, fig_loc] + ([fig_pho_] if chuyen_gia else [])
                + ([fig_dap_ung_] if fig_dap_ung_ is not None else []) + [fig_hr, fig_rr_])
    pdf = tao_pdf(thong_tin, kpi, dong_ss, luu_y,
                  dict(ecg=ecg_clean, rpeaks=rpeaks, fs=fs, t0=t_start, ref=ref), figs_pdf)
    st.download_button(t("⬇ Tải báo cáo PDF"), pdf, f"bao_cao_ecg_{ten.replace(' ', '_')}.pdf",
                       "application/pdf", type="primary", use_container_width=True)

plt.close("all")
