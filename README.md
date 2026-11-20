# 🫀 App phân tích tín hiệu ECG

Ứng dụng Streamlit phân tích tín hiệu điện tim (ECG) – Lab Tin học Vật lý Kỹ thuật.
Kết quả chỉ dùng cho học tập/nghiên cứu, **không dùng để chẩn đoán**.

## Dùng online (không cần cài đặt)

Mở link app trên Streamlit Community Cloud (xem mục *About* của repo này).
Mỗi khi code trên nhánh `main` thay đổi, app online tự cập nhật.

## Chức năng

- Đọc file CSV/TXT: có hoặc không có dòng tiêu đề, 1 hay nhiều cột, dấu `,` `;` hoặc tab
- Đơn vị mV, V, µV, hoặc giá trị ADC 12-bit (tự quy đổi theo giả định AD8232)
- Bộ lọc: NeuroKit, BioSPPy, Pan-Tompkins, Hamilton, hoặc **Tùy chỉnh**
  (Butterworth chỉnh bậc và tần số cắt, Notch 50/60 Hz, khử trôi nền và khử nhiễu bằng wavelet)
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

## Dữ liệu mẫu

`ecg_mau_500Hz.csv`: tín hiệu ECG mô phỏng 60 giây, tần số lấy mẫu 500 Hz (cột `time_s`, `ecg_mV`).
