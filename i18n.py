"""Song ngữ Việt/Anh. Chuỗi gốc tiếng Việt là khoá; t() trả bản tiếng Anh khi người dùng chọn English."""


def ngon_ngu() -> str:
    try:
        import streamlit as st
        return st.session_state.get("lang", "vi")
    except Exception:
        return "vi"


def t(s, /, **kw):
    if not isinstance(s, str):
        return s
    if ngon_ngu() == "en":
        s = EN.get(s, s)
    return s.format(**kw) if kw else s


EN = {}
EN.update({
    # --- khung chung
    "Chế độ chuyên gia": "Expert mode",
    "Bật để chỉnh chi tiết bộ lọc, xem phổ tần số và các thông số kỹ thuật.":
        "Turn on to fine-tune filters and see spectra and technical details.",
    "Phân tích điện tâm đồ": "ECG analysis",
    "Lab Tin học Vật lý Kỹ thuật · Kết quả chỉ dùng cho học tập/nghiên cứu, không dùng để chẩn đoán.":
        "Engineering Physics Informatics Lab · For teaching and research only, not for diagnosis.",
    "Nhập dữ liệu": "Load data", "Chọn bộ lọc": "Choose filter", "Phân tích": "Analyse",
    "Xem & xuất báo cáo": "Review & export report",
    "1. Dữ liệu": "1. Data", "Nguồn dữ liệu": "Data source", "Dùng dữ liệu mẫu": "Use sample data",
    "Tải file lên": "Upload files", "File dữ liệu ECG": "ECG data files",
    "CSV/TXT; EDF/EDF+ (.edf); Biopac (.acq); PhysioNet WFDB: chọn cùng lúc .hea + .dat, "
    "có thể kèm file nhãn .atr/.qrs/.ann để đối chiếu đỉnh R.":
        "CSV/TXT; EDF/EDF+ (.edf); Biopac (.acq); PhysioNet WFDB: select .hea + .dat together, optionally "
        "with an .atr/.qrs/.ann annotation file to check R-peaks.",
    "2. Thông tin bệnh nhân": "2. Patient", "Họ tên / mã": "Name / ID", "Bệnh nhân mẫu": "Sample patient",
    "Tuổi": "Age", "Giới tính": "Sex", "Nam": "Male", "Nữ": "Female",
    "3. Tín hiệu": "3. Signal", "Cột tín hiệu ECG": "ECG column", "Thiết bị đo": "Recording device",
    "Chọn máy đo để app tự điền tần số lấy mẫu và đơn vị. Thông tin ghi trong file luôn được ưu tiên.":
        "Pick the device to fill in sampling rate and units. Values stored in the file always take priority.",
    "Tần số ghi trong file được ưu tiên.": "The sampling rate stored in the file is used.",
    "Tần số lấy mẫu (Hz)": "Sampling rate (Hz)",
    "Đọc từ file {mt}: {fs} Hz, {n} kênh.": "Read from {mt}: {fs} Hz, {n} channel(s).",
    "Bỏ {k} kênh có tần số khác.": "{k} channel(s) with other rates skipped.",
    "File không có cột thời gian: hãy chọn **thiết bị đo** hoặc nhập **đúng tần số lấy mẫu**, "
    "nếu sai thì nhịp tim và mọi khoảng thời gian sẽ sai theo.":
        "The file has no time column: pick the **recording device** or enter the **correct sampling rate**. "
        "If it is wrong, heart rate and every interval will be wrong too.",
    "Đơn vị tín hiệu": "Signal unit", "Đơn vị ghi trong file: {u}": "Unit stored in file: {u}",
    "Đối tượng": "Subject", "Người lớn": "Adult", "Thai nhi (FECG)": "Fetus (FECG)",
    "Đổi ngưỡng đánh giá: người lớn 60–100 l/p, thai nhi 110–160 l/p (FIGO).":
        "Changes the reference: adult 60–100 bpm, fetus 110–160 bpm (FIGO).",
    "**Nhãn chuẩn:** {n} đỉnh R ({src})": "**Reference labels:** {n} R-peaks ({src})",
    "Dung sai khớp đỉnh R (ms)": "R-peak matching tolerance (ms)",
    "Chuẩn ANSI/AAMI EC57 dùng 150 ms cho người lớn; ECG thai nhi nhịp nhanh nên thường dùng 50 ms.":
        "ANSI/AAMI EC57 uses 150 ms for adults; fetal ECG is faster, so 50 ms is common.",
    "4. Bộ lọc": "4. Filter", "🧭 Đề xuất bộ lọc cho file này": "🧭 Suggest a filter for this file",
    "App tự thử mọi cách lọc trên file này và gợi ý cách tốt nhất (mất vài giây).":
        "Tries every filter on this file and suggests the best one (takes a few seconds).",
    "Phương pháp lọc": "Filter", "Thuật toán": "Algorithm",
    "Bật **Chế độ chuyên gia** để chỉnh từng bước lọc.": "Turn on **Expert mode** to adjust each filter step.",
    "▶ Phân tích": "▶ Analyse",
    # --- thiết bị, đơn vị
    "Không chọn (tự nhập)": "None (enter manually)",
    "Biopac MP36 / MP45 (Student Lab)": "Biopac MP36 / MP45 (Student Lab)",
    "BSL Lesson ECG mặc định 1000 Hz; file .acq đã ghi sẵn tần số.":
        "BSL ECG lessons default to 1000 Hz; .acq files store the rate.",
    "Biopac MP150 / MP160 (AcqKnowledge)": "Biopac MP150 / MP160 (AcqKnowledge)",
    "Thường ghi 500–2000 Hz; file .acq đã ghi sẵn tần số.": "Usually 500–2000 Hz; .acq files store the rate.",
    "AD8232 + ESP32 (ADC 12-bit, 3,3 V)": "AD8232 + ESP32 (12-bit ADC, 3.3 V)",
    "AD8232 + Arduino Uno/Nano (ADC 10-bit, 5 V)": "AD8232 + Arduino Uno/Nano (10-bit ADC, 5 V)",
    "Tần số phụ thuộc lệnh delay trong code: delay(4) ≈ 250 Hz, delay(5) ≈ 200 Hz.":
        "The rate depends on the delay in your code: delay(4) ≈ 250 Hz, delay(5) ≈ 200 Hz.",
    "MIT-BIH / Holter (PhysioNet)": "MIT-BIH / Holter (PhysioNet)", "MIT-BIH Arrhythmia ghi 360 Hz.":
        "MIT-BIH Arrhythmia is sampled at 360 Hz.",
    "Máy điện tim 12 chuyển đạo (PTB-XL)": "12-lead electrocardiograph (PTB-XL)",
    "PTB-XL có bản 500 Hz và 100 Hz.": "PTB-XL comes in 500 Hz and 100 Hz versions.",
    "Đai ngực Polar H10": "Polar H10 chest strap", "Polar H10 xuất ECG 130 Hz, đơn vị µV.":
        "Polar H10 streams ECG at 130 Hz in µV.",
    "V (đổi sang mV)": "V (convert to mV)", "µV (đổi sang mV)": "µV (convert to mV)",
    "ADC 12-bit → mV (giả định AD8232, 3,3 V, khuếch đại 1100)": "12-bit ADC → mV (assumes AD8232, 3.3 V, gain 1100)",
    "ADC 10-bit → mV (giả định AD8232, 5 V, khuếch đại 1100)": "10-bit ADC → mV (assumes AD8232, 5 V, gain 1100)",
    "ADC thô (chuẩn hoá, không có đơn vị)": "Raw ADC (normalised, no unit)",
    # --- tên phương pháp lọc
    "Tiêu chuẩn (mặc định)": "Standard (default)", "Giữ hình dạng sóng": "Preserves waveform",
    "Đếm nhịp – gọn nhẹ": "Beat counting – lightweight", "Đếm nhịp – ít nhiễu chen": "Beat counting – less noise",
    "Tự chỉnh": "Custom", "Tự chỉnh riêng cho file này": "Custom, tuned for this file",
    "Ổn định, dùng tốt cho hầu hết bản ghi.": "Robust; works well for most recordings.",
    "Đỉnh R thấp đi một chút sau khi lọc.": "R-peaks come out slightly lower after filtering.",
    "Sóng P, QRS, T rõ nét nhất. Nên chọn khi cần đo PR, QRS, QT.":
        "Clearest P, QRS and T waves. Choose it when you need PR, QRS, QT.",
    "Bản ghi nhiều nhiễu cơ thì đường sóng còn hơi gợn.": "With heavy muscle noise the trace stays a little rough.",
    "Thuật toán kinh điển, gọn nhẹ. Hợp để đếm nhịp và tính nhịp tim.":
        "Classic, lightweight algorithm. Good for counting beats and heart rate.",
    "Làm mất sóng P, T: không dùng để đo PR, QT.": "Removes P and T waves: do not use it for PR or QT.",
    "Dải lọc hẹp hơn nên ít nhiễu chen vào. Hợp để đếm nhịp.": "Narrower band lets in less noise. Good for counting beats.",
    "Tự chọn từng bước lọc, dành cho người cần chỉnh chi tiết.": "Set each filter step yourself, for detailed tuning.",
    "Chỉnh không hợp lý có thể làm méo sóng. Nên bấm Đề xuất trước.":
        "Poor settings can distort the waves. Try Suggest first.",
    "Bộ lọc thông thường": "Standard filter", "Wavelet (tốt khi nền trôi mạnh)": "Wavelet (better for strong drift)",
    "Mạnh (Universal)": "Strong (Universal)", "Nhẹ, giữ chi tiết (Bayes)": "Gentle, keeps detail (Bayes)",
    "Mềm (mượt hơn)": "Soft (smoother)", "Cứng (giữ biên độ)": "Hard (keeps amplitude)",
    "Từng mức": "Per level", "Một mức chung": "Single estimate",
    "**① Giữ dải tần của tim** · Butterworth": "**① Keep the cardiac band** · Butterworth",
    "Độ dứt khoát khi cắt (bậc)": "Cut-off sharpness (order)",
    "Cao hơn: cắt gọn hơn nhưng dễ tạo gợn quanh QRS. Bậc 2–4 là đủ cho ECG.":
        "Higher cuts more sharply but can ring around the QRS. Order 2–4 is enough for ECG.",
    "Bỏ phần dưới (Hz)": "Remove below (Hz)", "Bỏ phần trên (Hz)": "Remove above (Hz)",
    "0,5 Hz: làm phẳng đường nền. 0,05 Hz: giữ nguyên đoạn ST.": "0.5 Hz flattens the baseline. 0.05 Hz preserves the ST segment.",
    "40 Hz: chuẩn khi theo dõi, loại nhiễu cơ. Tối đa {m} Hz với file này.":
        "40 Hz is standard for monitoring and removes muscle noise. Max {m} Hz for this file.",
    "Số ở ô \"Bỏ phần dưới\" phải nhỏ hơn ô \"Bỏ phần trên\".": "\"Remove below\" must be lower than \"Remove above\".",
    "Làm phẳng đường nền bằng": "Flatten the baseline with",
    "Đường nền trôi lên xuống do thở hoặc cử động. Wavelet bám nền tốt hơn khi trôi mạnh.":
        "The baseline drifts with breathing or movement. Wavelet tracks strong drift better.",
    "**② Lọc nhiễu điện lưới** · Notch": "**② Remove mains noise** · Notch", "Bật lọc nhiễu điện": "Remove mains noise",
    "Loại nhiễu từ ổ điện, đèn, máy móc xung quanh.": "Removes interference from sockets, lights and nearby equipment.",
    "Tần số điện": "Mains frequency", "Việt Nam dùng điện 50 Hz.": "Vietnam and Europe use 50 Hz; the Americas 60 Hz.",
    "Độ hẹp (Q)": "Narrowness (Q)", "Lớn hơn: chỉ cắt đúng tần số điện, ít ảnh hưởng tín hiệu tim.":
        "Higher cuts only the mains frequency, with less effect on the cardiac signal.",
    "Lọc cả bội số (100, 150 Hz)": "Also remove harmonics (100, 150 Hz)",
    "**③ Làm mịn tín hiệu** · Wavelet": "**③ Smooth the signal** · Wavelet", "Bật làm mịn": "Smooth",
    "Giảm nhiễu cơ và nhiễu lăn tăn trên đường sóng.": "Reduces muscle noise and fine jitter on the trace.",
    "Dạng sóng mẫu": "Mother wavelet", "sym8 và db4 có hình giống phức bộ QRS nên hợp với ECG.":
        "sym8 and db4 resemble the QRS complex, so they suit ECG.",
    "Mức làm mịn": "Smoothing level", "Cao hơn: mịn hơn nhưng dễ mất sóng P, T. Mức 3–5 là hợp lý.":
        "Higher is smoother but may lose P and T waves. Levels 3–5 are sensible.",
    "Mức làm mịn này có thể làm mất sóng P và T. Nên giảm xuống.": "This level may remove P and T waves. Lower it.",
    "Độ mạnh": "Strength", "Kiểu làm mịn": "Smoothing type", "Ước lượng nhiễu": "Noise estimate",
    "giữ dải {lo}–{hi} Hz": "keeps {lo}–{hi} Hz", "làm phẳng nền bằng wavelet": "wavelet baseline removal",
    "lọc nhiễu điện {f} Hz": "{f} Hz mains notch", "làm mịn wavelet": "wavelet smoothing",
    # --- đọc file, lỗi
    "File rỗng.": "The file is empty.", "File không có cột số.": "The file has no numeric column.",
    "File không có kênh tín hiệu nào.": "The file has no signal channels.",
    "Cần cài thư viện bioread để đọc file Biopac: pip install bioread":
        "The bioread package is needed to read Biopac files: pip install bioread",
    "File Biopac này không có dữ liệu (file .gtl chỉ là mẫu cài đặt bài thí nghiệm). Hãy mở bản ghi trong Biopac "
    "Student Lab rồi lưu thành file .acq.":
        "This Biopac file contains no data (.gtl files are lesson templates). Open the recording in Biopac Student "
        "Lab and save it as an .acq file.",
    "nhãn EDF+ trong file": "EDF+ annotations in the file",
    "File .dat của PhysioNet cần tải lên kèm file .hea cùng tên.": "A PhysioNet .dat file must be uploaded with its .hea file.",
    "Chưa có file tín hiệu. Hãy tải .csv/.txt, .edf, .acq hoặc cặp .hea + .dat.":
        "No signal file yet. Upload .csv/.txt, .edf, .acq or a .hea + .dat pair.",
    "⬅️ Hãy tải file ECG lên ở thanh bên: CSV/TXT, EDF (.edf), Biopac (.acq), hoặc dữ liệu PhysioNet WFDB "
    "(.hea + .dat, kèm nhãn .atr/.qrs nếu có).":
        "⬅️ Upload an ECG file in the sidebar: CSV/TXT, EDF (.edf), Biopac (.acq), or PhysioNet WFDB "
        "(.hea + .dat, with .atr/.qrs labels if available).",
    "Không đọc được file: {e}": "Could not read the file: {e}", "Lỗi khi phân tích: {e}": "Analysis failed: {e}",
    "Tìm được quá ít đỉnh R. Kiểm tra lại cột dữ liệu và tần số lấy mẫu.":
        "Too few R-peaks found. Check the data column and the sampling rate.",
    # --- thanh bệnh nhân
    "Bệnh nhân": "Patient", "Tuổi / Giới": "Age / Sex", "Dữ liệu": "Data", "Kênh": "Channel", "Ghi": "Recording",
    "Ngày": "Date", "Dữ liệu mẫu (mô phỏng)": "Sample data (simulated)",
    "Xem 10 dòng đầu của dữ liệu": "Show the first 10 rows",
    # --- đề xuất
    "Đang phân tích phổ và chạy thử các phương pháp lọc...": "Analysing the spectrum and trying each filter...",
    "**Nên dùng: {ten}** · {d}/100 điểm": "**Recommended: {ten}** · {d}/100 points",
    "chủ yếu dựa trên mức khớp với nhãn đỉnh R của bác sĩ đi kèm file":
        "mainly reflects agreement with the expert R-peak labels in the file",
    "nhịp rõ và đều, bắt nhịp đúng, giữ hình dạng sóng, sạch nhiễu":
        "clear consistent beats, correct detection, waveform preserved, low noise",
    "App đã thử 5 cách lọc trên {s} giây đầu của bản ghi. Điểm {tc}.":
        "The app tried 5 filters on the first {s} seconds of the recording. The score {tc}.",
    "Hạng": "Rank", "Cách lọc": "Filter", "Điểm": "Score", "Nhận xét": "Comment",
    "Nhịp rõ, đều": "Clear, consistent beats", "Bắt nhịp đúng": "Correct detection", "Sạch nhiễu": "Low noise",
    "Khớp nhãn bác sĩ": "Agreement with expert labels",
    "**App nhận thấy gì về nhiễu của file này:**": "**What the app found about noise in this file:**",
    "Phương án áp dụng": "Option to apply", "✅ Áp dụng và phân tích": "✅ Apply and analyse",
    "Giữ dải {lo}–{hi} Hz: ": "Keep {lo}–{hi} Hz: ",
    "phức bộ QRS của thai nhi hẹp, cần giữ thêm tần số cao.": "the fetal QRS is narrow, so higher frequencies are kept.",
    "dải chuẩn khi theo dõi điện tim người lớn.": "the standard band for adult ECG monitoring.",
    "Đường nền trôi nhiều ({p}% năng lượng tín hiệu) → làm phẳng bằng wavelet.":
        "Strong baseline drift ({p}% of signal energy) → wavelet baseline removal.",
    "Đường nền trôi ít ({p}% năng lượng tín hiệu) → bộ lọc thông thường là đủ.":
        "Little baseline drift ({p}% of signal energy) → a standard filter is enough.",
    "Có nhiễu điện lưới {f} Hz rõ rệt → bật lọc nhiễu điện": "Clear {f} Hz mains noise → mains notch on",
    " (cả bội số 100, 150 Hz).": " (including 100 and 150 Hz harmonics).",
    "Không thấy nhiễu điện lưới → không cần lọc 50 Hz, tránh cắt bớt tín hiệu.":
        "No mains noise found → no 50 Hz notch, so no signal is cut.",
    "Có nhiễu cơ, nhiễu lăn tăn ({p}%) → bật làm mịn wavelet mức {m}.":
        "Muscle noise and jitter present ({p}%) → wavelet smoothing at level {m}.",
    "Ít nhiễu cơ ({p}%) → không cần làm mịn.": "Little muscle noise ({p}%) → no smoothing needed.",
    "Không chạy được trên file này.": "Could not run on this file.",
    "Tốt: sạch nhiễu mà vẫn giữ hình dạng sóng.": "Good: low noise with the waveform preserved.",
    "Lưu ý: ": "Note: ", "làm mất sóng P/T": "removes P/T waves", "bỏ sót hoặc bắt nhầm một số nhịp":
        "misses or mis-detects some beats", "các nhịp sau lọc không đồng đều": "filtered beats are inconsistent",
    "còn nhiều nhiễu": "noise remains",
    # --- kết quả
    "Bấm **▶ Phân tích** ở thanh bên để bắt đầu.": "Press **▶ Analyse** in the sidebar to start.",
    "Đang lọc tín hiệu, tìm đỉnh R và tính chỉ số...": "Filtering, detecting R-peaks and computing measures...",
    "**Nhịp tim {hr} l/p cao bất thường, hãy kiểm tra tần số lấy mẫu.** ":
        "**Heart rate {hr} bpm is unusually high; check the sampling rate.** ",
    "**Nhịp tim {hr} l/p thấp bất thường, hãy kiểm tra tần số lấy mẫu.** ":
        "**Heart rate {hr} bpm is unusually low; check the sampling rate.** ",
    "File không ghi tần số lấy mẫu nên app đang dùng **{fs} Hz** (theo thiết bị/ô nhập). Nếu con số này khác tần "
    "số thật của máy đo, nhịp tim và mọi khoảng thời gian đều sai theo cùng tỷ lệ.":
        "The file does not store its sampling rate, so the app is using **{fs} Hz** (from the device or input box). "
        "If the real rate differs, heart rate and every interval are off by the same ratio.",
    "Tần số lấy mẫu đang dùng có thể chưa đúng.": "The sampling rate in use may be wrong.",
    "Ước tính nhịp tim nếu tần số lấy mẫu thật là: ": "Estimated heart rate if the true sampling rate is: ",
    ". Hãy xem cấu hình máy đo (ví dụ chu kỳ đọc ADC trong code Arduino) để biết con số đúng.":
        ". Check the device settings (for example the ADC read interval in the Arduino code) for the right value.",
    "Biên độ mV được quy đổi từ giá trị ADC theo giả định AD8232 (khuếch đại 1100), chỉ mang tính ước lượng.":
        "mV amplitudes are converted from ADC values assuming an AD8232 (gain 1100) and are approximate.",
    "l/p": "bpm", "Nhịp tim trung bình": "Mean heart rate", "Thấp nhất / cao nhất": "Lowest / highest",
    "nhịp tức thời": "instantaneous", "Số nhịp": "Beats", "trong {s} s ghi": "in {s} s recorded",
    "Chất lượng tín hiệu": "Signal quality", "không đánh giá được": "not assessable",
    "Ngoài khoảng tham chiếu": "Outside reference range", "xem tab So sánh chỉ số": "see the Comparison tab",
    "tất cả trong khoảng": "all within range",
    "🎯 Đối chiếu với nhãn chuẩn": "🎯 Agreement with reference labels", "Độ nhạy Se": "Sensitivity Se",
    "Giá trị dự báo dương PPV": "Positive predictive value PPV", "Lệch thời gian TB": "Mean timing offset",
    "Tỷ lệ nhãn chuẩn được app tìm thấy = TP/(TP+FN)": "Share of reference beats found = TP/(TP+FN)",
    "Tỷ lệ đỉnh app tìm là đúng = TP/(TP+FP)": "Share of detected peaks that are correct = TP/(TP+FP)",
    "TP = {tp} · bỏ sót FN = {fn} · phát hiện thừa FP = {fp} · dung sai ±{ds} ms · nhãn: {src}.":
        "TP = {tp} · missed FN = {fn} · extra FP = {fp} · tolerance ±{ds} ms · labels: {src}.",
    "Kênh bụng mẹ (Abdomen) chứa cả ECG của mẹ (biên độ lớn) lẫn của thai nhi, nên thuật toán chủ yếu bắt nhịp "
    "**mẹ**. Nhãn chuẩn là nhịp **thai**, vì vậy Se/PPV thấp là đúng bản chất tín hiệu. Hãy chọn kênh **Direct_1** "
    "(điện cực đặt trên đầu thai nhi) để đánh giá thuật toán.":
        "Abdominal channels carry both the maternal ECG (large) and the fetal ECG, so the detector mostly finds "
        "**maternal** beats. The labels are **fetal** beats, so low Se/PPV is expected. Choose **Direct_1** (scalp "
        "electrode on the fetus) to evaluate the algorithm.",
    # --- tab
    "📈 Dải sóng ECG": "📈 ECG strip", "🧹 Trước/sau lọc": "🧹 Before/after filtering",
    "❤️ Nhịp tim & HRV": "❤️ Heart rate & HRV", "🩺 So sánh chỉ số": "🩺 Comparison", "📄 Báo cáo": "📄 Report",
    "Bắt đầu từ (s)": "Start at (s)", "Khung nhìn (s)": "View width (s)",
    "Kéo để dời theo thời gian · cuộn chuột hoặc kéo trên trục để phóng to · nhấp đúp để về khung ban đầu · rê "
    "chuột lên chấm xanh để xem nhịp tim tức thời.":
        "Drag to pan · scroll or drag on an axis to zoom · double-click to reset · hover a blue dot for the "
        "instantaneous heart rate.",
    "Tam giác đỏ: nhãn chuẩn.": "Red triangles: reference labels.",
    "Chưa cài plotly nên hiển thị dải sóng tĩnh.": "plotly is not installed, so a static strip is shown.",
    "Dải sóng chuẩn 25 mm/s, 10 mm/mV (bản in trong báo cáo)": "Standard strip 25 mm/s, 10 mm/mV (as printed in the report)",
    "Cách lọc đang dùng: {m}": "Filter in use: {m}",
    "Bật Chế độ chuyên gia để xem phổ tần số trước/sau lọc.": "Turn on Expert mode to see spectra before/after filtering.",
    "Đường cong cho biết dải tần được giữ lại (gần 0 dB) và dải bị loại bỏ. Phần làm mịn wavelet không thể hiện ở "
    "đây; xem hiệu quả của nó ở biểu đồ phổ phía trên.":
        "The curve shows which frequencies are kept (near 0 dB) and which are removed. Wavelet smoothing is not "
        "shown here; see its effect in the spectrum above.",
    "Tín hiệu ngắn hơn 30 s nên không tính HRV. Nên ghi tối thiểu 5 phút cho HRV ngắn hạn.":
        "The signal is shorter than 30 s, so HRV is not computed. Record at least 5 minutes for short-term HRV.",
    "Nhịp tim trung bình (l/p)": "Mean heart rate (bpm)", "Nhịp tim min / max (l/p)": "Heart rate min / max (bpm)",
    "RR trung bình (ms)": "Mean RR (ms)", "Đối chiếu nhãn chuẩn": "Reference labels",
    "Chỉ số": "Measure", "Giá trị": "Value", "Khoảng tham chiếu": "Reference range", "Đánh giá": "Assessment",
    "Độ tin cậy đo": "Confidence", "Đóng góp SHAP": "SHAP contribution", "Nguồn tham chiếu": "Source",
    "Phần này **không đưa ra chẩn đoán**. Mỗi chỉ số được đặt cạnh khoảng tham chiếu đã công bố để bác sĩ đánh "
    "giá nhanh; kết luận lâm sàng thuộc về bác sĩ.":
        "This section **does not make a diagnosis**. Each measure is placed next to a published reference range "
        "for quick review; the clinical conclusion belongs to the physician.",
    "Chỉ số trong khoảng tham chiếu": "Measures within reference", "Chỉ số ngoài khoảng": "Measures outside reference",
    "Điểm lệch tổng hợp": "Total deviation score",
    "0 = mọi chỉ số nằm trong khoảng tham chiếu. Không phải điểm nguy cơ.":
        "0 = every measure is within its reference range. This is not a risk score.",
    "**Chỉ số nào đóng góp vào mức lệch?** (biểu đồ thác nước kiểu SHAP)":
        "**Which measures contribute to the deviation?** (SHAP-style waterfall)",
    "Cách đọc biểu đồ SHAP và độ tin cậy": "How to read the SHAP chart and confidence",
    "- Biểu đồ bắt đầu từ **giá trị nền 0**, nghĩa là mọi chỉ số nằm trong khoảng tham chiếu. Mỗi thanh là phần "
    "đóng góp của một chỉ số; cộng dồn lại ra **điểm lệch tổng hợp**.\n- Đóng góp = mức lệch chuẩn hoá ra ngoài "
    "khoảng (1 = lệch thêm một đoạn bằng độ rộng khoảng, tối đa 2) × hệ số độ tin cậy của phép đo (Tốt 1,0 · Trung "
    "bình 0,7 · Kém 0,4). Các chỉ số có trọng số bằng nhau.\n- Vì điểm lệch là tổng của từng chỉ số (mô hình cộng "
    "tính), giá trị SHAP của mỗi chỉ số **đúng bằng** phần đóng góp của nó, không phải ước lượng như khi giải thích "
    "mô hình học máy hộp đen.\n- **Độ tin cậy đo** dựa trên chất lượng tín hiệu (SQI). Các khoảng PR/QRS/QTc đo từ "
    "hình thái sóng nên được hạ một bậc. HRV cần bản ghi ≥ 5 phút và ít nhịp lệch.\n- Đỏ = cao hơn khoảng tham "
    "chiếu, xanh dương = thấp hơn, xám = trong khoảng.":
        "- The chart starts from a **baseline of 0**, meaning every measure is within its reference range. Each bar "
        "is one measure's contribution; together they add up to the **total deviation score**.\n- Contribution = "
        "normalised distance outside the range (1 = one range-width beyond it, capped at 2) × measurement "
        "confidence factor (Good 1.0 · Fair 0.7 · Poor 0.4). All measures are weighted equally.\n- Because the score "
        "is a sum over measures (an additive model), each measure's SHAP value **equals** its contribution exactly; "
        "it is not an approximation as with black-box machine-learning models.\n- **Measurement confidence** comes "
        "from signal quality (SQI). PR/QRS/QTc are measured from wave morphology, so they are lowered one level. "
        "HRV needs a recording of at least 5 minutes with few irregular beats.\n- Red = above the range, blue = "
        "below, grey = within.",
    "**Bảng so sánh**": "**Comparison table**", "**Các điểm cần bác sĩ lưu ý**": "**Points for the physician to note**",
    "Tất cả chỉ số đánh giá được đều nằm trong khoảng tham chiếu.": "All assessable measures are within their reference ranges.",
    "Bản ghi thai nhi: chỉ so sánh nhịp tim thai cơ bản và biến thiên theo FIGO; không đánh giá PR/QRS/QT theo "
    "ngưỡng người lớn.":
        "Fetal recording: only baseline FHR and variability are compared (FIGO); PR/QRS/QT are not assessed "
        "against adult ranges.",
    "Khoảng tham chiếu đang dùng là của người lớn; trẻ em có ngưỡng khác theo lứa tuổi.":
        "Adult reference ranges are used; children have age-specific ranges.",
    "Nhịp số": "Beat no.", "Thời điểm R (s)": "R time (s)", "Nhịp tim (l/p)": "Heart rate (bpm)",
    "Số liệu đo chi tiết": "Detailed measurements", "⬇ Bảng so sánh (CSV)": "⬇ Comparison table (CSV)",
    "⬇ Số liệu đo (CSV)": "⬇ Measurements (CSV)", "⬇ Từng nhịp (CSV)": "⬇ Beat-by-beat (CSV)",
    "Tần số lấy mẫu": "Sampling rate", "Thời lượng ghi": "Recording length",
    "Trang đầu là **phiếu kết quả** khổ A4: thông tin bệnh nhân, ô chỉ số chính, dải sóng 10 giây, bảng so sánh "
    "với khoảng tham chiếu, các điểm cần lưu ý và ô để bác sĩ ghi kết luận, ký tên. Các trang sau là biểu đồ chi tiết.":
        "Page one is an A4 **result sheet**: patient details, key measures, a 10-second strip, the comparison with "
        "reference ranges, points to note and a box for the physician's conclusion and signature. Later pages hold "
        "the detailed charts.",
    "⬇ Tải báo cáo PDF": "⬇ Download PDF report",
    # --- trạng thái, chỉ số, nguồn
    "Cao": "High", "Thấp": "Low", "Bình thường": "Normal", "Không đánh giá": "Not assessed",
    "Tốt": "Good", "Trung bình": "Fair", "Kém": "Poor",
    "Nhịp tim": "Heart rate", "Nhịp tim TB": "Mean HR", "Nhịp tim thai cơ bản": "Baseline fetal HR",
    "Biến thiên nhịp tim thai": "Fetal HR variability",
    "FIGO 2015 (biên độ dao động trong 1 phút)": "FIGO 2015 (amplitude of oscillation per minute)",
    "AHA – nhịp xoang khi nghỉ": "AHA – resting sinus rhythm",
    "AHA/ACCF/HRS 2009 (≥ 110 ms: rối loạn dẫn truyền trong thất)":
        "AHA/ACCF/HRS 2009 (≥ 110 ms: intraventricular conduction delay)",
    "AHA/ACCF/HRS 2009 ({g} ≤ {q} ms)": "AHA/ACCF/HRS 2009 ({g} ≤ {q} ms)",
    "Nunan 2010 – HRV ngắn hạn 5 phút khi nghỉ": "Nunan 2010 – resting 5-minute short-term HRV",
    "{ten} = {x} {dv}, cao hơn khoảng tham chiếu {k} (độ tin cậy phép đo: {tc}).":
        "{ten} = {x} {dv}, above the reference range {k} (measurement confidence: {tc}).",
    "{ten} = {x} {dv}, thấp hơn khoảng tham chiếu {k} (độ tin cậy phép đo: {tc}).":
        "{ten} = {x} {dv}, below the reference range {k} (measurement confidence: {tc}).",
    # --- biểu đồ
    "Dải sóng ECG": "ECG strip", "biên độ phóng đại": "amplitude magnified",
    "chấm xanh: đỉnh R app tìm được": "blue dots: detected R-peaks", "tam giác đỏ: nhãn chuẩn": "red triangles: reference labels",
    "Thời gian (s)": "Time (s)", "Biên độ (mV)": "Amplitude (mV)", "Đỉnh R": "R-peak", "Nhãn chuẩn": "Reference label",
    "Tín hiệu gốc": "Raw signal", "Sau khi lọc": "Filtered", "Trước lọc": "Before", "Sau lọc": "After",
    "Tần số (Hz)": "Frequency (Hz)", "Mật độ công suất (mV²/Hz)": "Power density (mV²/Hz)",
    "Phổ công suất (Welch) trước và sau lọc": "Power spectrum (Welch) before and after filtering",
    "Biên độ (dB)": "Magnitude (dB)",
    "Đáp ứng tần số của bộ lọc (Butterworth lọc 2 chiều + Notch)": "Filter frequency response (zero-phase Butterworth + notch)",
    "Ngưỡng {a}–{b} l/p": "Range {a}–{b} bpm", "Nhịp tim tức thời theo thời gian": "Instantaneous heart rate over time",
    "Khoảng RR (ms)": "RR interval (ms)", "Phân bố khoảng RR": "RR interval distribution", "Biểu đồ Poincaré": "Poincaré plot",
    "Giá trị đo so với khoảng tham chiếu (dải xanh)": "Measured values against reference ranges (green band)",
    "thấp hơn": "below", "cận dưới": "lower limit", "cận trên": "upper limit", "cao hơn": "above",
    "không đo được": "not measurable", "Đóng góp vào điểm lệch (giá trị SHAP)": "Contribution to deviation (SHAP value)",
    "Mức đóng góp của từng chỉ số · giá trị nền E[f] = 0 → điểm lệch tổng hợp f(x) = {v}":
        "Contribution of each measure · baseline E[f] = 0 → total deviation f(x) = {v}",
    # --- báo cáo PDF
    "LAB TIN HỌC VẬT LÝ KỸ THUẬT": "ENGINEERING PHYSICS INFORMATICS LAB",
    "Trường Đại học Bách Khoa – ĐHQG TP.HCM": "Ho Chi Minh City University of Technology – VNU-HCM",
    "Mã phiếu": "Report no.", "PHIẾU KẾT QUẢ PHÂN TÍCH ĐIỆN TÂM ĐỒ": "ELECTROCARDIOGRAM ANALYSIS REPORT",
    "DẢI SÓNG ĐIỆN TIM 10 GIÂY": "10-SECOND ECG STRIP",
    "25 mm/s · lưới 0,2 s × 0,5 mV · chấm xanh: đỉnh R": "25 mm/s · grid 0.2 s × 0.5 mV · blue dots: R-peaks",
    "SO SÁNH VỚI KHOẢNG THAM CHIẾU": "COMPARISON WITH REFERENCE RANGES", "Kết quả": "Result",
    "CÁC ĐIỂM CẦN BÁC SĨ LƯU Ý": "POINTS FOR THE PHYSICIAN TO NOTE", "KẾT LUẬN CỦA BÁC SĨ": "PHYSICIAN'S CONCLUSION",
    "Ngày ..... tháng ..... năm 20.....": "Date ...... / ...... / 20......", "BÁC SĨ ĐỌC KẾT QUẢ": "REPORTING PHYSICIAN",
    "(Ký, ghi rõ họ tên)": "(Signature and full name)",
    "Kết quả tự động chỉ dùng cho học tập/nghiên cứu, không thay thế chẩn đoán y khoa.":
        "Automated results for teaching and research only; not a substitute for medical diagnosis.",
    "Trang": "Page",
})
