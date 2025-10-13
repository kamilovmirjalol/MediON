# ppg.py
# Lightweight PPG pipeline module for MicroPython:
# - Band-pass filter (0.5–4 Hz) tuned for MAX30102
# - Accel-referenced ANC (NLMS) to nibble motion
# - Beat detection for UI (independent of ML extractor)
# - Streaming feature engine (HR/HRV + morphology)
#
# Usage:
#   from ppg import PPGProcessor
#   ppg = PPGProcessor(fs=100)
#   out = ppg.process_sample(now_ms, ir_raw, acc_vert_dev_g, step_hz, acc_vrms)
#   # out: dict with keys: ir_filtered, bpm_avg, bpm_inst, hrv, beat_led_until
#   # feature engine: ppg.bfe

import math

# ---- High-resolution HR series config ----
HR_SERIES_HORIZON_S     = 60        # last 60 s
HR_SERIES_STEP_S        = 5         # sample every 5 s => 13 points
HR_SERIES_AVG_WIN_S     = 8         # trailing avg window to compute HR at each timestamp
HR_SHORT_HRV_S          = 30        # short HRV window for sdnn30/rmssd30

# UI: beat indicator
BEAT_LED_MS             = 140
MIN_INTERVAL_MS         = 300

# --- Original simple IIR band-pass that detector was tuned for ---
class BandPassFilter:
    def __init__(self, fs=100, low=0.5, high=4.0):
        self.alpha_hp = 1 / (1 + (fs / (2 * math.pi * low)))
        self.hp_prev_x = 0.0
        self.hp_prev_y = 0.0
        self.alpha_lp = (2 * math.pi * high) / (fs + 2 * math.pi * high)
        self.lp_prev_y = 0.0
    def filter(self, x):
        hp_y = self.alpha_hp * (self.hp_prev_y + x - self.hp_prev_x)
        self.hp_prev_x = x
        self.hp_prev_y = hp_y
        lp_y = self.alpha_lp * hp_y + (1 - self.alpha_lp) * self.lp_prev_y
        self.lp_prev_y = lp_y
        return lp_y

# --- Tiny helper one-pole filters for accel prefiltering ---
class OnePole:
    def __init__(self, fs, kind, fc):
        if kind == 'hp':
            self.a = 1.0 / (1.0 + (fs / (2 * math.pi * fc)))
            self.kind = 'hp'
            self.x1 = 0.0; self.y1 = 0.0
        else:
            self.a = (2 * math.pi * fc) / (fs + 2 * math.pi * fc)
            self.kind = 'lp'
            self.y1 = 0.0
            self.x1 = None
    def step(self, x):
        if self.kind == 'hp':
            y = self.a * (self.y1 + x - self.x1)
            self.y1 = y; self.x1 = x
            return y
        else:
            y = self.a * x + (1 - self.a) * self.y1
            self.y1 = y
            return y

# --- Small, guarded NLMS ANC that only nibbles motion (keeps waveform feel) ---
class NLMS_Small:
    def __init__(self, taps=8, mu=0.08, delta=1e-2, leak=1e-4, ref_delay=1):
        self.M = max(4, int(taps))
        self.mu = float(mu)
        self.delta = float(delta)
        self.leak = float(leak)
        self.w = [0.0]*self.M
        self.u = [0.0]*self.M
        self.ref_delay = max(0, int(ref_delay))
        self.delay_line = [0.0]*(self.ref_delay+1)
    def process(self, d, ref, adapt=True):
        self.delay_line.pop(0); self.delay_line.append(ref)
        r = self.delay_line[0]
        self.u[1:] = self.u[:-1]
        self.u[0] = r
        y_hat = 0.0
        for wi, ui in zip(self.w, self.u):
            y_hat += wi * ui
        e = d - y_hat  # cleaned signal
        if adapt:
            norm2 = 0.0
            for ui in self.u:
                norm2 += ui*ui
            g = self.mu / (norm2 + self.delta)
            for i in range(self.M):
                self.w[i] = (1 - self.leak)*self.w[i] + g*self.u[i]*e
        return e

# ------------ Streaming beat features for ML ------------
class BeatFeatureEngine:
    """
    Streaming extractor. Call `feed(now_ms, x)` for every cleaned PPG sample.
    Builds beat series and aggregates HR/HRV + morphology features.
    """
    def __init__(self, fs=100, window_min=10):
        self.fs = fs
        self.dt_ms = int(1000 // fs)
        self.win_ms = int(window_min * 60 * 1000)
        self.buf = []
        self.buf_max = int(2.0 * fs)  # ~2 s buffer
        self.bt_t = []         # beat timestamps (ms)
        self.bt_amp = []       # amplitude per beat
        self.bt_rise_ms = []   # rise time per beat
        self.bt_w50_ms = []    # width@50% per beat
        self._win = []
        self._last_peak_t = None
        self._pending_w50 = []
        self._k = 7
        self._rel_prom_th = 0.5
        self._min_rr_ms = 300
        self._max_rr_ms = 1500
        self._search_w50_ms = 800

    def _evict_window(self, now):
        cut = now - self.win_ms
        while self.bt_t and self.bt_t[0] < cut:
            self.bt_t.pop(0)
            if self.bt_amp: self.bt_amp.pop(0)
            if self.bt_rise_ms: self.bt_rise_ms.pop(0)
            if self.bt_w50_ms: self.bt_w50_ms.pop(0)

    def _register_peak(self, t_ms, peak_val):
        # estimate local minimum before peak (~0.6 s window)
        lookback = int(0.6 * self.fs)
        segment = self.buf[-lookback:] if lookback < len(self.buf) else self.buf[:]
        if segment:
            vmin = min(segment)
            imin = len(self.buf) - 1 - segment[::-1].index(vmin)
            t_min = t_ms - (len(self.buf) - 1 - imin) * self.dt_ms
        else:
            vmin = peak_val
            t_min = t_ms
        amp = peak_val - vmin
        rise_ms = max(0, t_ms - t_min)
        th = vmin + 0.5 * amp
        self._pending_w50.append({'th': th, 't_peak': t_ms, 't_cross': None})
        if self._last_peak_t is None:
            rr_ok = True
        else:
            rr = t_ms - self._last_peak_t
            rr_ok = (self._min_rr_ms <= rr <= self._max_rr_ms)
        self._last_peak_t = t_ms
        if rr_ok:
            self.bt_t.append(t_ms)
            self.bt_amp.append(float(amp))
            self.bt_rise_ms.append(int(rise_ms))

    def _update_w50(self, t_ms, x):
        keep = []
        for p in self._pending_w50:
            if p['t_cross'] is None:
                if x <= p['th']:
                    p['t_cross'] = t_ms
                    width = t_ms - p['t_peak']
                    if self.bt_t and self.bt_t[-1] == p['t_peak']:
                        self.bt_w50_ms.append(int(width))
                else:
                    if (t_ms - p['t_peak']) <= self._search_w50_ms:
                        keep.append(p)
        self._pending_w50 = keep

    def feed(self, now_ms, x):
        self.buf.append(x)
        if len(self.buf) > self.buf_max:
            self.buf.pop(0)
        self._win.append(x)
        if len(self._win) > self._k:
            self._win.pop(0)
        self._update_w50(now_ms, x)
        if len(self._win) == self._k:
            c = self._win[self._k // 2]
            if all(c >= v for v in self._win):
                prom = c - min(self._win)
                amp = max(self._win) - min(self._win)
                rel = (prom / amp) if amp > 0 else 0.0
                if rel >= self._rel_prom_th:
                    if (self._last_peak_t is None) or ((now_ms - self._last_peak_t) >= self._min_rr_ms):
                        self._register_peak(now_ms - (self._k // 2) * self.dt_ms, c)
        self._evict_window(now_ms)

    # ------- Utilities over beat timeline -------
    def _slice_by_time(self, now_ms, window_ms):
        tcut = now_ms - window_ms
        idx0 = 0
        while idx0 < len(self.bt_t) and self.bt_t[idx0] < tcut:
            idx0 += 1
        return idx0

    def _ibis_from(self, idx0):
        ts = self.bt_t[idx0:]
        return [ts[i] - ts[i-1] for i in range(1, len(ts))], ts

    def _hr_series(self, now_ms, horizon_s, step_s, avg_win_s):
        n_steps = int(horizon_s // step_s)
        xs = [-(horizon_s - i*step_s) for i in range(n_steps)] + [0]
        ys = []
        for x in xs:
            t_samp = now_ms + int(x * 1000)
            t0 = t_samp - int(avg_win_s * 1000)
            bpms = []
            for i in range(1, len(self.bt_t)):
                t_curr = self.bt_t[i]
                t_prev = self.bt_t[i-1]
                if t_prev >= t0 and t_curr <= t_samp and t_curr > t_prev:
                    rr = t_curr - t_prev
                    if 300 <= rr <= 1500:
                        bpms.append(60000.0 / rr)
            ys.append(sum(bpms)/len(bpms) if bpms else 0.0)
        return xs, ys

    def features(self, now_ms, window_ms=120000):
        idx0 = self._slice_by_time(now_ms, window_ms)
        ibis, ts = self._ibis_from(idx0)
        n = len(ts)
        if n < 3:
            xs, ys = self._hr_series(now_ms, HR_SERIES_HORIZON_S, HR_SERIES_STEP_S, HR_SERIES_AVG_WIN_S)
            return {
                'n_beats': n, 'hr_mean_bpm': 0.0,
                'sdnn_ms': 0.0, 'rmssd_ms': 0.0, 'pnn20': 0.0,
                'sd1_ms': 0.0, 'sd2_ms': 0.0,
                'amp_mean': 0.0, 'amp_std': 0.0,
                'rise_ms_mean': 0.0, 'width50_ms_mean': 0.0,
                'sqi': 0.0,
                'event_duration_s': 0.0,
                'hr_series': ys, 'hr_slope_60s': 0.0, 'hr_var_60s': 0.0,
                'sdnn30_ms': 0.0, 'rmssd30_ms': 0.0,
                'cv_ibi': 0.0, 'amp_cv': 0.0
            }

        mean_rr = sum(ibis) / len(ibis)
        var_rr = sum((rr - mean_rr) ** 2 for rr in ibis) / len(ibis)
        sdnn = math.sqrt(var_rr)
        diffs = [ibis[i] - ibis[i-1] for i in range(1, len(ibis))]
        rmssd = math.sqrt(sum(d*d for d in diffs) / len(diffs)) if diffs else 0.0
        pnn20 = (sum(1 for d in diffs if abs(d) > 20) / len(diffs)) if diffs else 0.0
        sd_diff = math.sqrt(sum((d - (sum(diffs)/len(diffs)))**2 for d in diffs) / len(diffs)) if diffs else 0.0
        sd1 = sd_diff / math.sqrt(2)
        sd2_sq = 2*(sdnn**2) - 0.5*(sd_diff**2)
        sd2 = math.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
        hr_mean = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        amps  = self.bt_amp[idx0:] if len(self.bt_amp)  >= len(self.bt_t) else []
        rises = self.bt_rise_ms[idx0:] if len(self.bt_rise_ms) >= len(self.bt_t) else []
        w50s  = self.bt_w50_ms[idx0:] if len(self.bt_w50_ms) >= len(self.bt_t) else []

        def mstats(arr):
            if not arr: return (0.0, 0.0)
            m = sum(arr)/len(arr)
            v = sum((a-m)**2 for a in arr)/len(arr)
            return (m, math.sqrt(v))

        amp_mean, amp_std = mstats(amps)
        rise_mean, _      = mstats(rises)
        w50_mean, _       = mstats(w50s)

        plausible = [1 if (300 <= rr <= 1500) else 0 for rr in ibis]
        sqi = (sum(plausible) / len(plausible)) if plausible else 0.0

        # Event duration
        event_duration_s = 0.0
        if len(ts) >= 2:
            event_duration_s = (ts[-1] - ts[0]) / 1000.0
            if event_duration_s > (120000/1000.0):
                event_duration_s = 120000/1000.0

        # High-resolution HR series over last 60 s
        xs, ys = self._hr_series(now_ms, HR_SERIES_HORIZON_S, HR_SERIES_STEP_S, HR_SERIES_AVG_WIN_S)
        xs_fit = []; ys_fit = []
        for x, y in zip(xs, ys):
            if y > 0:
                xs_fit.append(x); ys_fit.append(y)
        if len(xs_fit) >= 3:
            xmean = sum(xs_fit)/len(xs_fit)
            ymean = sum(ys_fit)/len(ys_fit)
            num = sum((x - xmean)*(y - ymean) for x, y in zip(xs_fit, ys_fit))
            den = sum((x - xmean)*(x - xmean) for x in xs_fit)
            hr_slope = (num / den) if den > 0 else 0.0
            hr_var = sum((y - ymean)**2 for y in ys_fit) / len(ys_fit)
        else:
            hr_slope = 0.0; hr_var = 0.0

        # Short-window HRV (30 s)
        idx30 = self._slice_by_time(now_ms, HR_SHORT_HRV_S*1000)
        ibis30, _ = self._ibis_from(idx30)
        if len(ibis30) >= 2:
            mean_rr30 = sum(ibis30)/len(ibis30)
            var_rr30 = sum((rr - mean_rr30)**2 for rr in ibis30)/len(ibis30)
            sdnn30 = math.sqrt(var_rr30)
            diffs30 = [ibis30[i] - ibis30[i-1] for i in range(1, len(ibis30))]
            rmssd30 = math.sqrt(sum(d*d for d in diffs30)/len(diffs30)) if diffs30 else 0.0
        else:
            sdnn30 = 0.0; rmssd30 = 0.0

        cv_ibi = (sdnn / mean_rr) if mean_rr > 0 else 0.0
        amp_cv = (amp_std / amp_mean) if amp_mean > 1e-9 else 0.0

        return {
            'n_beats': n,
            'hr_mean_bpm': hr_mean,
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'pnn20': pnn20,
            'sd1_ms': sd1,
            'sd2_ms': sd2,
            'amp_mean': amp_mean,
            'amp_std': amp_std,
            'rise_ms_mean': rise_mean,
            'width50_ms_mean': w50_mean,
            'sqi': sqi,
            'event_duration_s': event_duration_s,
            'hr_series': ys,
            'hr_slope_60s': hr_slope,
            'hr_var_60s': hr_var,
            'sdnn30_ms': sdnn30,
            'rmssd30_ms': rmssd30,
            'cv_ibi': cv_ibi,
            'amp_cv': amp_cv
        }

# ---- Helper to decide walking for ANC gating (mirrors your main thresholds) ----
def default_is_walking(step_hz, acc_vrms, walk_hz_min=0.7, walk_hz_max=2.3, walk_acc_max=0.22, run_enter_hz=2.9, acc_vert_enter=0.38):
    running = (step_hz >= run_enter_hz) or ((acc_vrms >= acc_vert_enter) and (step_hz >= 2.2))
    return (walk_hz_min <= step_hz <= walk_hz_max) and (acc_vrms <= walk_acc_max) and (not running)

class PPGProcessor:
    """
    Encapsulates the entire PPG processing chain + UI beat detection.
    """
    def __init__(self, fs=100, is_walking_fn=None):
        self.fs = fs
        self.bpf = BandPassFilter(fs=fs, low=0.5, high=4.0)
        self.hp = OnePole(fs, 'hp', 0.3)
        self.lp = OnePole(fs, 'lp', 8.0)
        self.anc = NLMS_Small(taps=8, mu=0.08, delta=1e-2, leak=1e-4, ref_delay=1)
        self.bfe = BeatFeatureEngine(fs=fs, window_min=10)
        self.is_walking_fn = is_walking_fn or default_is_walking

        # UI state
        self.signal_buffer = []
        self.buffer_size = 7
        self.last_beat_time = 0
        self.beat_led_until = 0
        self.bpm_history = []
        self.bpm_avg = 0.0
        self.bpm_inst = None
        self.rr_intervals = []
        self.HRV = 0.0

        # ANC RMS tracking
        self.acc_rms2 = 0.0
        self.acc_rms_alpha = 0.05

    def process_sample(self, now_ms, ir_raw, acc_vert_dev_g, step_hz, acc_vrms):
        # 1) IR band-pass
        ir_bp = self.bpf.filter(ir_raw)

        # 2) Build ANC reference from accel (HP->LP, RMS normalize)
        a_ref = self.lp.step(self.hp.step(acc_vert_dev_g))
        self.acc_rms2 = (1 - self.acc_rms_alpha)*self.acc_rms2 + self.acc_rms_alpha*(a_ref*a_ref)
        den = self.acc_rms2 if self.acc_rms2 > 1e-8 else 1e-8
        a_ref_n = a_ref / math.sqrt(den)

        # 3) Cleaned PPG
        adapt_ok = self.is_walking_fn(step_hz, acc_vrms) and (0.002 < den < 0.2)
        ir_filtered = self.anc.process(ir_bp, a_ref_n, adapt=adapt_ok)

        # 4) Feed to feature extractor
        self.bfe.feed(now_ms, ir_filtered)

        # 5) Simple beat detection for UI
        interval_ms = now_ms - self.last_beat_time
        self.signal_buffer.append(ir_filtered)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)
        if len(self.signal_buffer) == self.buffer_size:
            c = self.signal_buffer[self.buffer_size // 2]
            is_peak = all(c >= v for v in self.signal_buffer)
            prom = c - min(self.signal_buffer)
            amp = max(self.signal_buffer) - min(self.signal_buffer)
            rel = (prom / amp) if amp > 0 else 0.0
            if is_peak and interval_ms > MIN_INTERVAL_MS and rel > 0.5:
                self.last_beat_time = now_ms
                self.beat_led_until = now_ms + BEAT_LED_MS
                self.bpm_inst = 60000.0 / interval_ms
                if 40.0 < self.bpm_inst < 200.0:
                    self.bpm_history.append(self.bpm_inst)
                    if len(self.bpm_history) > 10: self.bpm_history.pop(0)
                    self.bpm_avg = sum(self.bpm_history) / len(self.bpm_history)
                    self.rr_intervals.append(interval_ms)
                    if len(self.rr_intervals) > 1:
                        mean_rr = sum(self.rr_intervals) / len(self.rr_intervals)
                        self.HRV = math.sqrt(sum((x - mean_rr) ** 2 for x in self.rr_intervals) / len(self.rr_intervals))
                    else:
                        self.HRV = 0.0

        return {
            'ir_filtered': ir_filtered,
            'bpm_avg': self.bpm_avg,
            'bpm_inst': self.bpm_inst,
            'hrv': self.HRV,
            'beat_led_until': self.beat_led_until
        }
