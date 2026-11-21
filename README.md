# 🫀 App phân tích tín hiệu ECG

Ứng dụng Streamlit phân tích tín hiệu điện tim (ECG) – Lab Tin học Vật lý Kỹ thuật.
Kết quả chỉ dùng cho học tập/nghiên cứu, **không dùng để chẩn đoán**.

## Dùng online (không cần cài đặt)

Mở link app trên Streamlit Community Cloud (xem mục *About* của repo này).
Mỗi khi code trên nhánh `main` thay đổi, app online tự cập nhật.

## Chức năng

- **Giao diện "phiếu lâm sàng"**: thanh thông tin bệnh nhân, thẻ chỉ số tô màu theo trạng thái (kèm đường nhịp tim nhỏ), thanh tiến trình 4 bước
- **Chế độ chuyên gia** (công tắc ở thanh bên): tắt = giao diện gọn cho bác sĩ; bật = chỉnh chi tiết bộ lọc, xem phổ, đáp ứng tần số
- **Song ngữ Tiếng Việt / English** cho toàn bộ giao diện, biểu đồ và báo cáo PDF
- Đọc file **Biopac** (`.acq`, AcqKnowledge / Biopac Student Lab)
- **Chọn thiết bị đo** (Biopac MP36/MP45/MP150, AD8232 + ESP32/Arduino, MIT-BIH, PTB-XL, Polar H10) để tự điền tần số lấy mẫu và đơn vị
- **Dải sóng tương tác** (Plotly): kéo để dời, cuộn để phóng to, rê chuột xem biên độ và nhịp tim tức thời, lưới giấy điện tim chuẩn
- **Báo cáo PDF khổ A4 kiểu phiếu kết quả bệnh viện**: đầu phiếu, thông tin hành chính, ô chỉ số chính, dải sóng 10 giây, bảng so sánh, ô kết luận và chữ ký bác sĩ

- Đọc file CSV/TXT: có hoặc không có dòng tiêu đề, 1 hay nhiều cột, dấu `,` `;` hoặc tab
- Đọc dữ liệu PhysioNet: **EDF/EDF+** (`.edf`) và **WFDB** (chọn cùng lúc `.hea` + `.dat`, định dạng 16/61/80/212)
- Đọc nhãn đỉnh R chuẩn (`.atr`, `.qrs`, `.ann` hoặc nhãn EDF+) và **đối chiếu**: độ nhạy Se, PPV, F1
- Chế độ **thai nhi (FECG)** với ngưỡng nhịp tim 110–160 l/p (FIGO)
- Đơn vị mV, V, µV, hoặc giá trị ADC 12-bit / 10-bit (tự quy đổi theo giả định AD8232)
- Bộ lọc: NeuroKit, BioSPPy, Pan-Tompkins, Hamilton, hoặc **Tùy chỉnh**
  (Butterworth chỉnh bậc và tần số cắt, Notch 50/60 Hz, khử trôi nền và khử nhiễu bằng wavelet)
- **Đề xuất bộ lọc cho từng file**: phân tích phổ (trôi nền, nhiễu 50/60 Hz, nhiễu tần cao) để tự sinh cấu hình lọc riêng, chạy thử mọi phương pháp, chấm điểm (chất lượng tín hiệu, nhịp hợp lý, giữ hình dạng sóng, khử nhiễu; hoặc F1 nếu có nhãn chuẩn) và áp dụng phương án tốt nhất bằng một nút bấm
- **So sánh với khoảng tham chiếu cho bác sĩ** (AHA/ACCF/HRS, Nunan 2010, FIGO): biểu đồ dạng phiếu xét nghiệm, biểu đồ thác nước **SHAP** cho biết chỉ số nào đóng góp vào mức lệch, độ tin cậy từng phép đo — không đưa ra chẩn đoán
- Tìm đỉnh R, nhịp tim, PR, QRS, QT, QTc (Bazett, Fridericia), HRV (SDNN, RMSSD, pNN50, SD1/SD2)
- Dải sóng trên lưới giấy điện tim chuẩn (25 mm/s, 10 mm/mV), phổ công suất, đáp ứng tần số bộ lọc
- Xuất bảng chỉ số (CSV) và báo cáo PDF

## Chạy trên máy (Windows)

Tải repo về (*Code → Download ZIP*), giải nén, nhấp đúp **`CHAY_APP.bat`**.
Lần đầu, script tự cài Python 3.12 và thư viện (mất vài phút), sau đó mở app tại http://localhost:8601.

## Chạy trên máy (macOS / Linux)

```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt -c constraints.txt
streamlit run app.py
```

## Cấu trúc mã nguồn

| File | Nội dung |
|---|---|
| `app.py` | Giao diện Streamlit |
| `xu_ly.py` | Đọc file, quy đổi đơn vị, thiết bị đo, bộ lọc, tìm đỉnh R, PR/QRS/QT, HRV |
| `bieu_do.py` | Biểu đồ (matplotlib, Plotly) và báo cáo PDF A4 |
| `so_sanh_chi_so.py` | So sánh với khoảng tham chiếu, biểu đồ SHAP |
| `de_xuat_loc.py` | Đề xuất bộ lọc cho từng file |
| `doc_physionet.py` | Đọc EDF/EDF+, WFDB và nhãn WFDB |
| `i18n.py` | Song ngữ Việt/Anh |
| `.streamlit/config.toml` | Màu giao diện |

## Dữ liệu mẫu

`ecg_mau_500Hz.csv`: tín hiệu ECG mô phỏng 60 giây, tần số lấy mẫu 500 Hz (cột `time_s`, `ecg_mV`).
