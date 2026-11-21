"""
Đọc dữ liệu PhysioNet không cần thư viện ngoài (chỉ numpy):
  - EDF / EDF+ (.edf), kể cả kênh "EDF Annotations"
  - WFDB tín hiệu (.hea + .dat, định dạng 16, 212, 80, 61)
  - WFDB nhãn (.atr, .qrs, .ann ... định dạng MIT)
"""
from __future__ import annotations

import re

import numpy as np

# Mã nhịp tim (beat) theo chuẩn WFDB: N, L, R, a, V, F, J, A, S, E, j, /, Q, e, n, r, f
MA_NHIP = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 30, 34, 35, 38, 41}
KY_HIEU = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E",
           11: "j", 12: "/", 13: "Q", 25: "e", 30: "n", 34: "r", 35: "f", 38: "?", 41: "B"}


# ----------------------------------------------------------------------------
# EDF / EDF+
# ----------------------------------------------------------------------------
def doc_edf(b: bytes) -> dict:
    """Trả về dict: signals {tên: mảng vật lý}, units, fs {tên: Hz}, ann_s (mốc thời gian nhãn EDF+, giây)."""
    def s(a, n):
        return b[a:a + n].decode("latin-1").strip()

    hdr_bytes = int(s(184, 8))
    n_rec = int(s(236, 8))
    dur = float(s(244, 8)) or 1.0
    ns = int(s(252, 4))
    o = 256

    def truong(width):
        nonlocal o
        vals = [s(o + i * width, width) for i in range(ns)]
        o += ns * width
        return vals

    labels = truong(16)
    truong(80)                       # transducer
    units = truong(8)
    pmin = np.array(truong(8), float)
    pmax = np.array(truong(8), float)
    dmin = np.array(truong(8), float)
    dmax = np.array(truong(8), float)
    truong(80)                       # prefiltering
    nsamp = np.array(truong(8), int)

    rec_len = int(nsamp.sum())
    data = np.frombuffer(b, dtype="<i2", offset=hdr_bytes)
    if n_rec < 0:                    # -1 = không rõ, tự suy ra
        n_rec = len(data) // rec_len
    data = data[: n_rec * rec_len].reshape(n_rec, rec_len)

    signals, unit_map, fs_map, ann_s = {}, {}, {}, []
    start = 0
    for i in range(ns):
        block = data[:, start:start + nsamp[i]]
        start += nsamp[i]
        if labels[i] == "EDF Annotations":
            ann_s += _doc_tal(block.tobytes())
            continue
        k = (pmax[i] - pmin[i]) / ((dmax[i] - dmin[i]) or 1)
        signals[labels[i]] = (block.reshape(-1).astype(float) - dmin[i]) * k + pmin[i]
        unit_map[labels[i]] = units[i]
        fs_map[labels[i]] = nsamp[i] / dur
    return dict(signals=signals, units=unit_map, fs=fs_map, ann_s=sorted(ann_s))


def _doc_tal(raw: bytes) -> list[float]:
    """Đọc Time-stamped Annotation Lists của EDF+; bỏ các mốc giữ thời gian (không có chữ)."""
    out = []
    for tal in raw.split(b"\x00"):
        if not tal:
            continue
        parts = tal.split(b"\x14")
        m = re.match(rb"([+-][0-9.]+)", parts[0])
        if m and any(p.strip() for p in parts[1:]):
            out.append(float(m.group(1)))
    return out


# ----------------------------------------------------------------------------
# WFDB tín hiệu (.hea + .dat)
# ----------------------------------------------------------------------------
def doc_wfdb(hea_text: str, dat_files: dict[str, bytes]) -> dict:
    lines = [l for l in hea_text.splitlines() if l.strip() and not l.startswith("#")]
    rec = lines[0].split()
    n_sig = int(rec[1])
    fs = float(rec[2].split("/")[0].split("(")[0]) if len(rec) > 2 else 250.0
    sigs = []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        fmt = p[1].split("x")[0].split(":")[0].split("+")[0]
        gain, base, unit = 200.0, None, "mV"
        if len(p) > 2:
            m = re.match(r"([0-9.eE+-]+)(?:\(([-0-9]+)\))?(?:/(\S+))?", p[2])
            if m:
                gain = float(m.group(1)) or 200.0
                base = int(m.group(2)) if m.group(2) else None
                unit = m.group(3) or "mV"
        adczero = int(p[4]) if len(p) > 4 else 0
        name = " ".join(p[8:]) if len(p) > 8 else f"kenh_{len(sigs) + 1}"
        sigs.append(dict(file=p[0], fmt=fmt, gain=gain, base=adczero if base is None else base,
                         unit=unit, name=name))

    signals, units = {}, {}
    for f in dict.fromkeys(s["file"] for s in sigs):
        if f not in dat_files:
            raise ValueError(f"Thiếu file dữ liệu '{f}' đi kèm file .hea.")
        group = [s for s in sigs if s["file"] == f]
        d = _giai_ma(dat_files[f], group[0]["fmt"], len(group))
        for j, s in enumerate(group):
            ten = s["name"] if s["name"] not in signals else f"{s['name']}_{j + 1}"
            signals[ten] = (d[:, j].astype(float) - s["base"]) / s["gain"]
            units[ten] = s["unit"]
    return dict(signals=signals, units=units, fs={k: fs for k in signals}, ann_s=[])


def _giai_ma(b: bytes, fmt: str, n: int) -> np.ndarray:
    if fmt == "16":
        d = np.frombuffer(b[: len(b) // (2 * n) * 2 * n], "<i2")
    elif fmt == "61":
        d = np.frombuffer(b[: len(b) // (2 * n) * 2 * n], ">i2")
    elif fmt == "80":
        d = np.frombuffer(b, np.uint8).astype(np.int16) - 128
    elif fmt == "212":
        a = np.frombuffer(b[: len(b) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int16)
        s1 = a[:, 0] | ((a[:, 1] & 0x0F) << 8)
        s2 = a[:, 2] | ((a[:, 1] & 0xF0) << 4)
        d = np.empty(2 * len(a), np.int16)
        d[0::2], d[1::2] = s1, s2
        d[d > 2047] -= 4096
    else:
        raise ValueError(f"Chưa hỗ trợ định dạng WFDB {fmt} (hỗ trợ 16, 61, 80, 212).")
    d = d[: len(d) // n * n]
    return d.reshape(-1, n)


# ----------------------------------------------------------------------------
# WFDB nhãn (MIT format): .atr, .qrs, .ann ...
# ----------------------------------------------------------------------------
def doc_nhan_wfdb(b: bytes) -> tuple[np.ndarray, list[str]]:
    """Trả về (vị trí mẫu của các nhịp, ký hiệu nhịp). Chỉ giữ các mã là nhịp tim."""
    w = np.frombuffer(b[: len(b) // 2 * 2], "<u2")
    t, i, mau, ky = 0, 0, [], []
    while i < len(w):
        a, inc = w[i] >> 10, int(w[i] & 0x3FF)
        i += 1
        if a == 0 and inc == 0:
            break
        if a == 59:                       # SKIP: 32-bit, word cao trước
            if i + 1 < len(w):
                v = (int(w[i]) << 16) | int(w[i + 1])
                t += v - (1 << 32) if v >= (1 << 31) else v
            i += 2
        elif a == 63:                     # AUX: bỏ qua chuỗi phụ
            i += (inc + 1) // 2
        elif a in (60, 61, 62):           # NUM, SUB, CHN
            continue
        else:
            t += inc
            if a in MA_NHIP:
                mau.append(t)
                ky.append(KY_HIEU.get(int(a), "?"))
    return np.asarray(mau, int), ky
