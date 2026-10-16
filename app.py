# -*- coding: utf-8 -*-
r"""
Студийный микрофон — обработка в реальном времени на твоём железе.
Scarlett 2i2 (Rode NT1) -> [Гейт -> Хай-пасс -> Эквалайзер -> Де-эссер ->
Компрессор -> Лимитер -> Гейн] -> Виртуальный микрофон (CABLE) -> Discord/OBS/игры.
Запуск: venv\Scripts\python.exe app.py  -> http://localhost:5002
"""
import os, time, threading, json
import numpy as np

print("Запускаю студийный микрофон... подожди пару секунд.", flush=True)

from flask import Flask, request, jsonify, send_file
from pedalboard import (Pedalboard, NoiseGate, HighpassFilter, LowShelfFilter,
                        PeakFilter, HighShelfFilter, Compressor, Limiter, Gain)
from pedalboard.io import AudioStream
import sounddevice as sd

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ── Цепочка обработки (все блоки — проф-DSP из Pedalboard) ─────────────────────
gate    = NoiseGate(threshold_db=-45, ratio=4, attack_ms=1, release_ms=120)
hpf     = HighpassFilter(cutoff_frequency_hz=85)          # убрать гул/бубнёж/поп
eq_low  = LowShelfFilter(cutoff_frequency_hz=120, gain_db=0.0, q=0.7)  # низ/тело
eq_mud  = PeakFilter(cutoff_frequency_hz=300, gain_db=0.0, q=1.0)      # «муть» 250-400
eq_pres = PeakFilter(cutoff_frequency_hz=4500, gain_db=0.0, q=1.0)     # чёткость/присутствие
eq_air  = HighShelfFilter(cutoff_frequency_hz=11000, gain_db=0.0, q=0.7)  # «воздух»
deess   = PeakFilter(cutoff_frequency_hz=7000, gain_db=0.0, q=2.5)     # приручить «с/ш»
comp    = Compressor(threshold_db=-18, ratio=3.0, attack_ms=5, release_ms=120)
limiter = Limiter(threshold_db=-1.5, release_ms=100)
outgain = Gain(gain_db=0.0)

board = Pedalboard([gate, hpf, eq_low, eq_mud, eq_pres, eq_air, deess, comp, limiter, outgain])

# имя параметра -> (объект, атрибут)
PARAMS = {
    "gate_threshold": (gate, "threshold_db"),
    "gate_ratio":     (gate, "ratio"),
    "hpf":            (hpf, "cutoff_frequency_hz"),
    "low_gain":       (eq_low, "gain_db"),
    "mud_freq":       (eq_mud, "cutoff_frequency_hz"),
    "mud_gain":       (eq_mud, "gain_db"),
    "pres_freq":      (eq_pres, "cutoff_frequency_hz"),
    "pres_gain":      (eq_pres, "gain_db"),
    "air_gain":       (eq_air, "gain_db"),
    "deess_gain":     (deess, "gain_db"),
    "comp_threshold": (comp, "threshold_db"),
    "comp_ratio":     (comp, "ratio"),
    "out_gain":       (outgain, "gain_db"),
}

def _get_state():
    return {k: float(getattr(o, a)) for k, (o, a) in PARAMS.items()}

# ── Сохранение/загрузка настроек (чтобы помнило после перезапуска) ─────────────
CONFIG = os.path.join(BASE, "config.json")
_sel = {"input": None, "output": None}

def save_config():
    try:
        json.dump({"params": _get_state(), "input": _sel["input"], "output": _sel["output"],
                   "running": bool(_status.get("running")), "denoise": float(_denoise.strength)},
                  open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass

def load_config():
    try:
        return json.load(open(CONFIG, encoding="utf-8"))
    except Exception:
        return {}

def autostart_from_config(force_start=True):
    """Применить сохранённые настройки и (по умолчанию) сразу запустить обработку —
    чтобы микрофон обрабатывался всегда, без открытия сайта."""
    cfg = load_config()
    if cfg.get("params"):
        _apply(cfg["params"])
    _denoise.strength = 0.0     # AI-шумодав убран из аудио-пути
    ins, outs = _inputs(), _outputs()
    # строгий поиск: НЕ падать на «первое попавшееся» устройство (иначе мик уйдёт в колонки)
    def _find(names, *needles):
        for n in names:
            for nd in needles:
                if nd.lower() in n.lower():
                    return n
        return None
    in_name = cfg.get("input") or _find(ins, "ocusrite", "scarlett", "2i2")
    out_name = cfg.get("output") or _find(outs, "CABLE Input", "CABLE In")
    _sel["input"], _sel["output"] = in_name, out_name
    if in_name and out_name and (force_start or cfg.get("running")):
        _start(in_name, out_name)
    save_config()

# ── Пресеты ───────────────────────────────────────────────────────────────────
PRESETS = {
    "nt1_broadcast": {   # Rode NT1 + Scarlett 2i2 — чистый вещательный голос
        "gate_threshold": -42, "gate_ratio": 4, "hpf": 90,
        "low_gain": 0, "mud_freq": 300, "mud_gain": -2.5,
        "pres_freq": 4500, "pres_gain": 3.0, "air_gain": 2.5,
        "deess_gain": -3.0, "comp_threshold": -18, "comp_ratio": 3.0, "out_gain": 3.0,
    },
    "warm_podcast": {    # тёплый, «радийный»
        "gate_threshold": -45, "gate_ratio": 4, "hpf": 80,
        "low_gain": 2.0, "mud_freq": 350, "mud_gain": -1.5,
        "pres_freq": 3500, "pres_gain": 2.0, "air_gain": 1.0,
        "deess_gain": -2.0, "comp_threshold": -20, "comp_ratio": 3.5, "out_gain": 4.0,
    },
    "flat": {            # нейтрально (почти без обработки)
        "gate_threshold": -60, "gate_ratio": 1.5, "hpf": 40,
        "low_gain": 0, "mud_freq": 300, "mud_gain": 0,
        "pres_freq": 4500, "pres_gain": 0, "air_gain": 0,
        "deess_gain": 0, "comp_threshold": -6, "comp_ratio": 1.5, "out_gain": 0,
    },
}

def _apply(values):
    for k, v in values.items():
        if k in PARAMS:
            o, a = PARAMS[k]
            setattr(o, a, float(v))
    if _stream is not None:
        try: _stream.plugins = board       # протолкнуть изменения в живой поток
        except Exception: pass

# ── Аудио-поток (Pedalboard AudioStream) ──────────────────────────────────────
_stream = None
_keep = False
_thread = None
_status = {"running": False, "error": "", "input": "", "output": ""}

# DSP-движок (двухпоточный) — включается, когда шумодав > 0
import soxr
from denoise import SpectralDenoiser
_denoise = SpectralDenoiser(hop=512)
_buf = {"d": np.zeros(0, dtype=np.float32)}   # накопитель обработанных на 48к
_ring_lock = threading.Lock()
_in_stream = None
_out_stream = None
_dsp = {"on": False}

def _pick(names, *needles):
    for n in names:
        for nd in needles:
            if nd.lower() in n.lower():
                return n
    return names[0] if names else None

def _inputs():  return list(dict.fromkeys(AudioStream.input_device_names))
def _outputs(): return list(dict.fromkeys(AudioStream.output_device_names))

def _audio_loop(in_name, out_name):
    global _stream, _keep
    try:
        with AudioStream(input_device_name=in_name, output_device_name=out_name,
                         plugins=board, allow_feedback=True, buffer_size=2048) as s:
            _stream = s
            _status.update({"running": True, "error": "", "input": in_name, "output": out_name})
            while _keep:
                time.sleep(0.1)
    except Exception as e:
        _status.update({"running": False, "error": str(e)})
    finally:
        _stream = None
        _status["running"] = False

def _start_stream(in_name, out_name):
    """Движок №1 — Pedalboard AudioStream (когда шумодав выключен). Идеальный, низкая задержка."""
    global _keep, _thread
    _keep = True
    _thread = threading.Thread(target=_audio_loop, args=(in_name, out_name), daemon=True)
    _thread.start()
    for _ in range(30):
        time.sleep(0.1)
        if _status["running"] or _status["error"]:
            break

def _stop_stream():
    global _keep, _thread
    _keep = False
    if _thread is not None:
        _thread.join(timeout=2)
        _thread = None

def _find_sd_output(namehint):
    apis = sd.query_hostapis()
    pref = sorted(range(len(apis)), key=lambda i: 0 if 'WASAPI' in apis[i]['name'] else (1 if 'MME' in apis[i]['name'] else 2))
    key = (namehint or "cable input")[:16].lower()
    for api in pref:
        for di in apis[api]['devices']:
            d = sd.query_devices(di)
            if d['max_output_channels'] > 0 and (key in d['name'].lower() or 'cable input' in d['name'].lower()):
                return di
    return None

def _start_dsp(in_name, out_name):
    """Движок №2 — свой (шумодав + вся цепочка). WASAPI (низкая задержка) + пересчёт
    частоты входа под выход через soxr (напр. Focusrite 192к -> CABLE 48к)."""
    global _in_stream, _out_stream
    in_idx = _find_sd_input(in_name)
    out_idx = _find_sd_output(out_name)
    if in_idx is None or out_idx is None:
        _status.update({"running": False, "error": "нет устройства для шумодава"}); return
    sr_in = int(sd.query_devices(in_idx)['default_samplerate'])
    sr_out = int(sd.query_devices(out_idx)['default_samplerate'])
    in_ch = int(sd.query_devices(in_idx)['max_input_channels'])   # WASAPI требует точное число каналов
    resampler = soxr.ResampleStream(sr_in, sr_out, 1, dtype='float32') if sr_in != sr_out else None
    in_block = max(256, int(round(512 * sr_in / sr_out)))   # чтобы на выходе ~512 сэмплов на колбэк
    _denoise.reset()
    with _ring_lock:
        _buf["d"] = np.zeros(0, dtype=np.float32)

    def in_cb(indata, frames, tinfo, status):
        x = indata[:, 0].astype(np.float32).copy()
        pk = float(np.abs(x).max()) if x.size else 0.0
        _level["peak"] = 20.0 * np.log10(pk + 1e-9); _level["ok"] = True
        if resampler is not None:
            x = resampler.resample_chunk(x)
        if x.size:
            with _ring_lock:
                _buf["d"] = np.concatenate([_buf["d"], x])

    def out_cb(outdata, frames, tinfo, status):
        with _ring_lock:
            if _buf["d"].shape[0] >= frames:
                x = _buf["d"][:frames]; _buf["d"] = _buf["d"][frames:]
            else:
                x = None
        if x is None:
            x = np.zeros(frames, dtype=np.float32)
        y = _denoise.process(x)                                  # шумодав
        try:
            y = np.asarray(board(y, sr_out, reset=False)).reshape(-1).astype(np.float32)  # цепочка
        except Exception:
            pass
        if y.shape[0] != frames:
            z = np.zeros(frames, dtype=np.float32); n = min(frames, y.shape[0]); z[:n] = y[:n]; y = z
        outdata[:] = np.repeat(y.reshape(-1, 1), outdata.shape[1], axis=1)

    try:
        _in_stream = sd.InputStream(device=in_idx, channels=in_ch, samplerate=sr_in,
                                    blocksize=in_block, dtype='float32', callback=in_cb)
        _out_stream = sd.OutputStream(device=out_idx, channels=2, samplerate=sr_out,
                                      blocksize=512, dtype='float32', callback=out_cb)
        _in_stream.start(); _out_stream.start()
        _dsp["on"] = True
        _status.update({"running": True, "error": "", "input": in_name, "output": out_name})
    except Exception as e:
        _status.update({"running": False, "error": str(e)})
        _stop_dsp()

def _stop_dsp():
    global _in_stream, _out_stream
    for s in (_in_stream, _out_stream):
        try:
            if s: s.stop(); s.close()
        except Exception:
            pass
    _in_stream = None; _out_stream = None
    _dsp["on"] = False
    if not _keep:
        _status["running"] = False

def _start(in_name, out_name):
    """Только чистый движок AudioStream (AI-шумодав убран — звук как раньше)."""
    _stop()
    _denoise.strength = 0.0
    _start_stream(in_name, out_name)
    _stop_meter(); _start_meter(in_name)

def _stop():
    _stop_stream()
    _stop_dsp()
    _stop_meter()
    _status["running"] = False

# ── Метр уровня входа (отдельный лёгкий поток на микрофоне) ────────────────────
_level = {"peak": -90.0, "ok": False}
_meter_stream = None
def _start_meter(in_name):
    global _meter_stream
    if _dsp.get("on"):
        return                      # в DSP-режиме метр считается прямо в потоке
    try:
        # ищем индекс устройства с этим (обрезанным) именем в sounddevice
        idx = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and in_name[:20].lower() in d["name"].lower():
                idx = i; break
        if idx is None:
            return
        sr = int(sd.query_devices(idx)["default_samplerate"])
        def cb(indata, frames, t, status):
            pk = float(np.abs(indata).max()) if indata.size else 0.0
            _level["peak"] = 20*np.log10(pk + 1e-9)
        _meter_stream = sd.InputStream(device=idx, channels=1, samplerate=sr,
                                       blocksize=1024, callback=cb)
        _meter_stream.start()
        _level["ok"] = True
    except Exception:
        _level["ok"] = False

def _stop_meter():
    global _meter_stream
    if _meter_stream is not None:
        try: _meter_stream.stop(); _meter_stream.close()
        except Exception: pass
        _meter_stream = None
    _level.update({"peak": -90.0, "ok": False})

# ── 🎯 Автонастройка под голос (анализ речи -> оптимальные параметры) ───────────
def _find_sd_input(namehint):
    apis = sd.query_hostapis()
    pref = sorted(range(len(apis)), key=lambda i: 0 if 'WASAPI' in apis[i]['name'] else (1 if 'MME' in apis[i]['name'] else 2))
    key = (namehint or "focusrite")[:18].lower()
    for api in pref:
        for di in apis[api]['devices']:
            d = sd.query_devices(di)
            if d['max_input_channels'] > 0 and (key in d['name'].lower() or 'focusrite' in d['name'].lower()):
                return di
    return None

def _band_db(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    p = psd[m].mean() if m.any() else 1e-12
    return 10.0 * np.log10(p + 1e-12)

def _analyze_and_apply(y, sr):
    y = np.asarray(y, dtype=np.float64)
    if y.size < sr * 2:
        return {"ok": False, "error": "Мало звука — попробуй ещё раз."}
    N, H = 2048, 1024
    idxs = range(0, len(y) - N, H)
    frames = [y[i:i+N] for i in idxs]
    if len(frames) < 8:
        return {"ok": False, "error": "Мало звука — говори дольше."}
    win = np.hanning(N)
    rms = np.array([np.sqrt(np.mean(f**2)) + 1e-9 for f in frames])
    rms_db = 20.0 * np.log10(rms)
    noise_db = float(np.percentile(rms_db, 10))
    speech_db = float(np.percentile(rms_db, 85))
    if speech_db - noise_db < 6:
        return {"ok": False, "error": "Не услышал голос — говори обычным голосом ближе к микрофону."}
    thr = noise_db + (speech_db - noise_db) * 0.5
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    psd = np.zeros(len(freqs)); cnt = 0
    for f, db in zip(frames, rms_db):
        if db > thr:
            psd += np.abs(np.fft.rfft(f * win))**2; cnt += 1
    if cnt < 3:
        return {"ok": False, "error": "Мало речи — попробуй ещё раз."}
    psd /= cnt
    mud  = _band_db(freqs, psd, 200, 500)
    body = _band_db(freqs, psd, 500, 1500)
    pres = _band_db(freqs, psd, 3000, 6000)
    sib  = _band_db(freqs, psd, 6000, 9000)
    air  = _band_db(freqs, psd, 9000, 15000)
    mud_gain  = float(np.clip(-(mud - body - 3) * 0.5, -6, 0))
    pres_gain = float(np.clip((body - pres) * 0.4, 0, 6))
    air_gain  = float(np.clip((body - air - 6) * 0.3, 0, 5))
    deess_gain= float(np.clip(-(sib - pres) * 0.5, -8, 0))
    gate_threshold = float(np.clip(noise_db + 8, -60, -24))
    comp_threshold = float(np.clip(speech_db - 6, -36, -8))
    out_gain = float(np.clip(-14 - speech_db, -3, 18))
    params = dict(gate_threshold=round(gate_threshold, 1), gate_ratio=10.0, hpf=90.0,
                  low_gain=0.0, mud_freq=300.0, mud_gain=round(mud_gain, 1),
                  pres_freq=4500.0, pres_gain=round(pres_gain, 1), air_gain=round(air_gain, 1),
                  deess_gain=round(deess_gain, 1), comp_threshold=round(comp_threshold, 1),
                  comp_ratio=3.0, out_gain=round(out_gain, 1))
    _apply(params); save_config()
    ex = [f"Фон {noise_db:.0f} dB → шумодав {gate_threshold:.0f} dB.",
          f"Голос {speech_db:.0f} dB → громкость {out_gain:+.0f} dB, компрессор {comp_threshold:.0f} dB.",
          (f"Убрал густоту {mud_gain:+.1f} dB @300Гц." if mud_gain < -0.4 else "Низ в норме."),
          (f"Добавил чёткости {pres_gain:+.1f} dB." if pres_gain > 0.4 else "Чёткость ок."),
          (f"Добавил воздуха {air_gain:+.1f} dB." if air_gain > 0.4 else "Верх ок."),
          (f"Приручил «с/ш» {deess_gain:+.1f} dB." if deess_gain < -0.4 else "Сибилянты ок.")]
    return {"ok": True, "params": _get_state(), "explain": " ".join(ex)}

def _calibrate(seconds=12):
    idx = _find_sd_input(_sel.get("input") or "")
    if idx is None:
        return {"ok": False, "error": "Нет входного устройства (включи обработку)."}
    sr = int(sd.query_devices(idx)['default_samplerate'])
    maxch = int(sd.query_devices(idx)['max_input_channels'])
    was = _level.get("ok")
    _stop_meter()
    mono, err = None, None
    for ch in [c for c in (2, maxch, 1) if 1 <= c <= maxch]:
        try:
            rec = sd.rec(int(seconds * sr), samplerate=sr, channels=ch, device=idx, dtype='float32')
            sd.wait()
            mono = rec[:, 0]        # 1-й вход = твой NT1
            break
        except Exception as e:
            err = e; continue
    if was: _start_meter(_sel.get("input"))
    if mono is None:
        return {"ok": False, "error": f"Запись не удалась: {err}"}
    return _analyze_and_apply(mono, sr)

# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))

@app.route("/devices")
def devices():
    ins, outs = _inputs(), _outputs()
    return jsonify({
        "inputs": ins, "outputs": outs,
        "suggest_in": _pick(ins, "ocusrite", "scarlett", "2i2"),
        "suggest_out": _pick(outs, "CABLE Input"),
        "monitor": _pick(outs, "Logitech", "headset", "наушник"),
    })

@app.route("/state")
def state():
    return jsonify({"params": _get_state(), "status": _status, "presets": list(PRESETS.keys()),
                    "denoise": float(_denoise.strength)})

@app.route("/levels")
def levels():
    return jsonify(_level)

@app.route("/set", methods=["POST"])
def set_param():
    d = request.json or {}
    _apply({d.get("param"): d.get("value")})
    save_config()
    return jsonify({"ok": True})

@app.route("/preset", methods=["POST"])
def preset():
    d = request.json or {}
    name = d.get("name")
    if name in PRESETS:
        _apply(PRESETS[name])
        save_config()
        return jsonify({"ok": True, "params": _get_state()})
    return jsonify({"ok": False, "error": "нет такого пресета"})

@app.route("/denoise", methods=["POST"])
def denoise_set():
    _denoise.strength = 0.0     # AI-шумодав убран — всегда чистый звук
    save_config()
    return jsonify({"ok": True, "strength": 0.0, "running": _status.get("running"), "engine": "stream"})

@app.route("/calibrate", methods=["POST"])
def calibrate():
    d = request.json or {}
    sec = max(6, min(int(d.get("seconds", 12)), 20))
    return jsonify(_calibrate(sec))

@app.route("/start", methods=["POST"])
def start():
    d = request.json or {}
    ins, outs = _inputs(), _outputs()
    in_name = d.get("input") or _pick(ins, "ocusrite", "scarlett", "2i2")
    out_name = d.get("output") or _pick(outs, "CABLE Input")
    if not in_name or not out_name:
        return jsonify({"ok": False, "error": "не нашёл устройства"})
    _sel["input"], _sel["output"] = in_name, out_name
    _start(in_name, out_name)
    save_config()
    return jsonify({"ok": _status["running"], "status": _status})

@app.route("/stop", methods=["POST"])
def stop():
    _stop(); save_config()
    return jsonify({"ok": True, "status": _status})

if __name__ == "__main__":
    import webbrowser
    def _open():
        import urllib.request
        for _ in range(120):
            try:
                urllib.request.urlopen("http://localhost:5002/", timeout=1)
                webbrowser.open("http://localhost:5002"); return
            except Exception:
                time.sleep(0.5)
    threading.Thread(target=_open, daemon=True).start()
    print("Открой в браузере: http://localhost:5002", flush=True)
    app.run(host="127.0.0.1", port=5002, threaded=True)
