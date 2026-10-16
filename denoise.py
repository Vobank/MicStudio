# -*- coding: utf-8 -*-
r"""
Спектральный шумодав в реальном времени (без внешних библиотек, только numpy).
Метод: STFT + винеровская маска + слежение за уровнем шума, 50% overlap-add
(окно Ханна) — низкая задержка (~один хоп), плавно, без «вкл/выкл» как у гейта.
strength: 0.0 = выкл (звук не трогаем), 1.0 = максимум подавления.
"""
import numpy as np


class SpectralDenoiser:
    def __init__(self, hop=512):
        self.hop = int(hop)
        self.frame = self.hop * 2
        self.win = np.hanning(self.frame).astype(np.float32)
        self.prev_in = np.zeros(self.hop, dtype=np.float32)   # предыдущий блок (для кадра)
        self.prev_out = np.zeros(self.hop, dtype=np.float32)  # хвост overlap-add
        self.noise = None                                     # оценка спектра шума (по бинам)
        self.strength = 0.0
        self.floor = 0.06                                     # мин. усиление (~-24 дБ), чтобы не «булькало»

    def reset(self):
        self.prev_in[:] = 0; self.prev_out[:] = 0; self.noise = None

    def process(self, block):
        """block: моно float32 длиной hop. Возвращает обработанный блок той же длины
        (с задержкой в один hop). При strength<=0 — почти прозрачно."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.shape[0] != self.hop:                        # подстраховка по длине
            b = np.zeros(self.hop, dtype=np.float32)
            n = min(self.hop, block.shape[0]); b[:n] = block[:n]; block = b

        s = float(self.strength)
        if s <= 0.001:
            # прозрачный путь, но держим состояние кадров синхронно
            out = np.concatenate([self.prev_in, block])[:self.hop]
            self.prev_in = block.copy()
            return out

        frame = np.concatenate([self.prev_in, block])         # frame = 2*hop
        self.prev_in = block.copy()

        X = np.fft.rfft(frame * self.win)
        mag = np.abs(X).astype(np.float32)
        eps = 1e-8

        if self.noise is None:
            self.noise = mag.copy()
        else:
            # слежение за шумом: тянемся к минимуму (шум тише речи)
            up = 0.98 * self.noise + 0.02 * mag
            self.noise = np.minimum(mag, up)
            # но даём шуму медленно расти, если кругом стабильно шумно
            self.noise = 0.995 * self.noise + 0.005 * mag

        snr = (mag * mag) / (self.noise * self.noise + eps)
        gain = snr / (snr + 1.0)                              # винеровская маска (0..1)
        gain = np.maximum(gain, self.floor)
        gain = 1.0 - s * (1.0 - gain)                         # регулировка силы

        y = np.fft.irfft(X * gain).astype(np.float32)
        y *= self.win                                         # синтез-окно
        out = y[:self.hop] + self.prev_out
        self.prev_out = y[self.hop:]
        return out
