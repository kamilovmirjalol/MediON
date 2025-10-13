"""
sensors.py
----------
Lightweight sensor abstraction for the MediON stress logger runtime.

Provides MAX30102 (PPG) and MPU6050 (IMU) access with simple buffering and
feature extraction that works across slightly different driver APIs.
"""

import math
import time

try:
    from max30102 import MAX30102
except ImportError:  # pragma: no cover
    MAX30102 = None  # type: ignore

try:
    from mpu6050 import mpu6050 as MPU6050Driver
except ImportError:  # pragma: no cover
    MPU6050Driver = None  # type: ignore


PPG_WINDOW_MS = 60_000
ACC_WINDOW_MS = 60_000
PPG_REFRACTORY_MS = 300
PPG_MIN_THRESH = 120  # raw units, tuned empirically


def _ticks_ms() -> int:
    try:
        return time.ticks_ms()
    except AttributeError:  # pragma: no cover
        return int(time.time() * 1000)


class RingBuffer:
    """Simple fixed-duration time-value buffer."""

    def __init__(self, max_ms: int):
        self.max_ms = max_ms
        self.times = []
        self.values = []

    def append(self, ts_ms: int, value):
        self.times.append(ts_ms)
        self.values.append(value)
        self._evict(ts_ms)

    def _evict(self, now_ms: int):
        while self.times and (now_ms - self.times[0]) > self.max_ms:
            self.times.pop(0)
            self.values.pop(0)

    def __len__(self):
        return len(self.times)

    def items(self):
        return zip(self.times, self.values)

    def copy_values(self):
        return list(self.values)

    def copy_times(self):
        return list(self.times)


class Sensors:
    """
    Wraps access to MAX30102 (PPG) and MPU6050 (IMU). Designed to stay resilient
    even when hardware is missing or uses a different driver signature.
    """

    def __init__(self, i2c):
        self.i2c = i2c
        self.ppg = None
        self.imu = None
        self.has_ppg = False
        self.has_imu = False
        self._last_ppg_error = None
        self._last_imu_error = None

        # Buffers
        self.ppg_buffer = RingBuffer(PPG_WINDOW_MS)
        self.acc_buffer = RingBuffer(ACC_WINDOW_MS)
        self.beat_times = []

        # Beat detection state
        self._dc_est = 0.0
        self._ac_est = 0.0
        self._last_signal = 0.0
        self._last_derivative = 0.0
        self._last_peak_val = None
        self._last_beat_ms = -PPG_REFRACTORY_MS

        self._init_sensors()

    # --------------------------------------------------------------------- init
    def _init_sensors(self):
        # ---- MAX30102 ----
        if MAX30102 is not None:
            # Prefer shared I2C instance to avoid bus conflicts
            try:
                self.ppg = MAX30102(i2c=self.i2c)
                self.has_ppg = True
            except TypeError:
                # Driver lacking i2c kw — try known pins on I2C0
                last_exc = None
                pinners = [(0, 1), (16, 17), (8, 9)]
                for sda_pin, scl_pin in pinners:
                    try:
                        self.ppg = MAX30102(i2c_bus=0, sda=sda_pin, scl=scl_pin)
                        self.has_ppg = True
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        self.ppg = None
                        self.has_ppg = False
                if last_exc is not None and not self.has_ppg:
                    self._last_ppg_error = last_exc
            except Exception as exc:
                self._last_ppg_error = exc
                self.ppg = None
                self.has_ppg = False

        # ---- MPU6050 ----
        if MPU6050Driver is not None:
            try:
                # Ensure device appears on the bus before init
                try:
                    addrs = self.i2c.scan()
                except Exception:
                    addrs = []
                if 0x68 not in addrs and 104 not in addrs:
                    raise OSError("MPU6050 not found on I2C bus")
                self.imu = MPU6050Driver(self.i2c)
                self.has_imu = True
            except Exception as exc:
                # One retry after short delay (power-up settling)
                try:
                    time.sleep_ms(80)
                except Exception:
                    pass
                try:
                    self.imu = MPU6050Driver(self.i2c)
                    self.has_imu = True
                except Exception as exc2:
                    self._last_imu_error = exc2
                    self.imu = None
                    self.has_imu = False

    # ------------------------------------------------------------------- helpers
    def read_ppg_sample(self):
        """Drain one IR sample from the MAX30102 FIFO (if available)."""
        if not self.has_ppg or self.ppg is None:
            return None

        available = 0
        try:
            if hasattr(self.ppg, "get_data_present"):
                available = self.ppg.get_data_present()
            elif hasattr(self.ppg, "check") and hasattr(self.ppg, "available"):
                # SparkFun-style driver
                self.ppg.check()
                available = 1 if self.ppg.available() else 0
            else:
                available = 1  # best effort: assume at least one sample
        except Exception as exc:
            self._last_ppg_error = exc
            return None

        if available <= 0:
            return None

        try:
            if hasattr(self.ppg, "read_fifo"):
                _, ir = self.ppg.read_fifo()
                return ir
            elif hasattr(self.ppg, "pop_ir_from_storage"):
                ir = self.ppg.pop_ir_from_storage()
                return ir
        except Exception as exc:
            self._last_ppg_error = exc
            return None

        return None

    def feed_ppg(self, sample):
        """Insert IR sample into buffers and update beat detector."""
        if sample is None:
            return
        now = _ticks_ms()
        self.ppg_buffer.append(now, int(sample))

        # Baseline and AC estimate (simple leaky integrator)
        alpha_dc = 0.01
        self._dc_est = (1.0 - alpha_dc) * self._dc_est + alpha_dc * sample
        signal = sample - self._dc_est

        alpha_ac = 0.1
        self._ac_est = (1.0 - alpha_ac) * self._ac_est + alpha_ac * abs(signal)

        derivative = signal - self._last_signal
        thr = max(PPG_MIN_THRESH, self._ac_est * 0.6)

        # Peak detection: slope change + refractory period.
        if (
            signal > thr
            and self._last_derivative > 0
            and derivative <= 0
            and (now - self._last_beat_ms) >= PPG_REFRACTORY_MS
        ):
            self._register_beat(now, signal)

        self._last_signal = signal
        self._last_derivative = derivative

    def _register_beat(self, ts_ms: int, peak_val: float):
        self.beat_times.append(ts_ms)
        # keep only last 60 seconds of beats
        cutoff = ts_ms - PPG_WINDOW_MS
        while self.beat_times and self.beat_times[0] < cutoff:
            self.beat_times.pop(0)
        self._last_peak_val = peak_val
        self._last_beat_ms = ts_ms

    def read_acc_sample(self):
        """Fetch (ax, ay, az) in g units if available."""
        if not self.has_imu or self.imu is None:
            return None

        try:
            if hasattr(self.imu, "get_accel_data"):
                data = self.imu.get_accel_data(g=True)
            elif hasattr(self.imu, "read_accel_data"):
                data = self.imu.read_accel_data(g=True)
            else:
                return None
        except Exception as exc:
            self._last_imu_error = exc
            return None

        try:
            ax, ay, az = float(data["x"]), float(data["y"]), float(data["z"])
        except Exception:
            return None
        return ax, ay, az

    def feed_acc(self, ax: float, ay: float, az: float):
        if ax is None or ay is None or az is None:
            return
        now = _ticks_ms()
        self.acc_buffer.append(now, (ax, ay, az))

    # ---------------------------------------------------------------- features
    def get_ppg_features(self):
        """
        Compute HR/PPG features on the last 60 seconds.
        Returns a dict with defaults if insufficient data.
        """
        now = _ticks_ms()
        self.ppg_buffer._evict(now)

        beats = [bt for bt in self.beat_times if (now - bt) <= PPG_WINDOW_MS]
        beat_count = len(beats)
        result = {
            "hr_mean": 0.0,
            "hr_std": 0.0,
            "rr_rmssd": 0.0,
            "acdc_ratio": 0.0,
            "ppg_sqi": 0.0,
            "beat_count": beat_count,
        }

        if beat_count < 2 or len(self.ppg_buffer) < 10:
            return result

        # RR intervals & HR series
        rr_intervals = []
        hr_series = []
        for i in range(1, beat_count):
            rr = beats[i] - beats[i - 1]
            if 300 <= rr <= 2000:
                rr_intervals.append(rr)
                hr_series.append(60000.0 / rr)

        if not rr_intervals:
            return result

        hr_mean = sum(hr_series) / len(hr_series)
        hr_std = math.sqrt(
            sum((h - hr_mean) ** 2 for h in hr_series) / len(hr_series)
        ) if len(hr_series) > 1 else 0.0
        rr_rmssd = math.sqrt(
            sum((rr_intervals[i] - rr_intervals[i - 1]) ** 2 for i in range(1, len(rr_intervals)))
            / (len(rr_intervals) - 1)
        ) if len(rr_intervals) > 1 else 0.0

        # AC/DC ratio using windowed samples
        samples = self.ppg_buffer.copy_values()
        if samples:
            dc = sum(samples) / len(samples)
            ac = max(samples) - min(samples)
            acdc_ratio = (ac / dc) if dc else 0.0
        else:
            acdc_ratio = 0.0

        # Simple SQI: fraction of 5 s bins with a beat + HR plausibility
        bins = 12  # 60 s / 5 s
        bin_ms = PPG_WINDOW_MS // bins
        coverage_set = set()
        if beats:
            window_start = now - PPG_WINDOW_MS
            for bt in beats:
                bin_idx = int((bt - window_start) // bin_ms)
                if 0 <= bin_idx < bins:
                    coverage_set.add(bin_idx)
        coverage_frac = len(coverage_set) / bins if bins else 0.0
        hr_ok = 1.0 if 40 <= hr_mean <= 180 else 0.0
        sqi = min(1.0, coverage_frac * 0.7 + hr_ok * 0.3)

        result.update(
            hr_mean=hr_mean,
            hr_std=hr_std,
            rr_rmssd=rr_rmssd,
            acdc_ratio=acdc_ratio,
            ppg_sqi=sqi,
        )
        return result

    def get_motion_features(self, motion_thresh_g: float):
        now = _ticks_ms()
        self.acc_buffer._evict(now)
        values = self.acc_buffer.copy_values()
        result = {
            "acc_var": 0.0,
            "motion_frac": 0.0,
            "sample_count": len(values),
        }
        if not values:
            return result

        mags = [math.sqrt(ax * ax + ay * ay + az * az) for ax, ay, az in values]
        mean_mag = sum(mags) / len(mags)
        var_mag = sum((m - mean_mag) ** 2 for m in mags) / len(mags)
        std_mag = math.sqrt(var_mag)

        threshold = max(motion_thresh_g, mean_mag + std_mag)

        # Estimate fraction of 1 s windows over threshold
        if mags:
            bin_ms = 1000
            times = self.acc_buffer.copy_times()
            start = times[0]
            bin_end = start + bin_ms
            total_bins = 0
            exceed_bins = 0
            idx = 0
            while start < times[-1]:
                total_bins += 1
                over = False
                while idx < len(times) and times[idx] < bin_end:
                    if mags[idx] > threshold:
                        over = True
                    idx += 1
                if over:
                    exceed_bins += 1
                start = bin_end
                bin_end = start + bin_ms
            motion_frac = exceed_bins / total_bins if total_bins else 0.0
        else:
            motion_frac = 0.0

        result.update(acc_var=var_mag, motion_frac=motion_frac)
        return result


__all__ = ["Sensors"]
