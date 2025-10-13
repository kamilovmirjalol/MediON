"""
main.py
-------
MediON stress data collection runtime for Raspberry Pi Pico / Pico W.

Collects PPG (MAX30102) and accelerometer (MPU6050) samples, derives features
in 60 s windows (with 30 s stride), logs CSV rows, and optionally evaluates a
compact Random Forest model to trigger DFPlayer alerts. Label inputs come from
an active-low button or console commands.
"""

import json
import math
import sys
import time

try:
    import uselect
except ImportError:
    uselect = None

from machine import I2C, Pin, UART

from sensors import Sensors
from logging_utils import (
    CSVLogger,
    isoformat_from_epoch,
    isoformat_from_uptime,
)
from rf_runtime import RandomForestRuntime

try:
    from dfplayer import DFPlayer
except ImportError:  # pragma: no cover
    DFPlayer = None  # type: ignore

# -------------------------------------------------------------------- constants
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "ALERT_THRESH": 0.6,
    "MIN_ALERT_GAP_SEC": 120,
    "MOTION_THRESH_G": 0.15,
    "BASELINE": {
        "hr_mean": None,
        "hr_std": None,
        "rr_rmssd": None,
        "rr_rmssd_std": None,
    },
}

WINDOW_MS = 60_000
WINDOW_STEP_MS = 30_000
LABEL_ACTIVE_MS = 120_000
PPG_POLL_MS = 20
ACC_POLL_MS = 20
BUTTON_PIN = 14
BUTTON_DEBOUNCE_MS = 40
BUTTON_LONG_MS = 1500

UART_ID = 1
UART_TX = 4
UART_RX = 5
DFPLAYER_BUSY_PIN = 17


# --------------------------------------------------------------------- utilities
def ticks_ms():
    try:
        return time.ticks_ms()
    except AttributeError:  # pragma: no cover
        return int(time.time() * 1000)


def ticks_diff(a, b):
    try:
        return time.ticks_diff(a, b)
    except AttributeError:  # pragma: no cover
        return a - b


def ticks_add(value, delta):
    try:
        return time.ticks_add(value, delta)
    except AttributeError:  # pragma: no cover
        return value + delta


def rtc_available():
    try:
        return time.localtime()[0] >= 2020
    except Exception:
        return False


def default_config():
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_config():
    try:
        with open(CONFIG_PATH) as fp:
            cfg = json.load(fp)
    except OSError:
        cfg = default_config()
        save_config(cfg)
        return cfg
    except ValueError:
        cfg = default_config()
        save_config(cfg)
        return cfg

    # Ensure defaults
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = default_config()[key] if isinstance(value, dict) else value
    if "BASELINE" not in cfg or not isinstance(cfg["BASELINE"], dict):
        cfg["BASELINE"] = default_config()["BASELINE"]
    else:
        for k, v in DEFAULT_CONFIG["BASELINE"].items():
            cfg["BASELINE"].setdefault(k, v)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as fp:
            json.dump(cfg, fp)
    except OSError:
        pass


def init_dfplayer():
    if DFPlayer is None:
        return None
    try:
        player = DFPlayer(uartInstance=UART_ID, txPin=UART_TX, rxPin=UART_RX, busyPin=DFPLAYER_BUSY_PIN)
        return player
    except Exception:
        return None


def play_alert(dfplayer, volume=24):
    if dfplayer is None:
        return False
    try:
        if hasattr(dfplayer, "setVolume"):
            dfplayer.setVolume(volume)
        busy = dfplayer.queryBusy() if hasattr(dfplayer, "queryBusy") else False
        if busy:
            return False
        dfplayer.playRoot(1)
        return True
    except Exception:
        return False


def make_feature_vector(ppg_feats, motion_feats, hour_sin, hour_cos, hr_z, rr_z):
    return {
        "hr_mean": ppg_feats["hr_mean"],
        "hr_std": ppg_feats["hr_std"],
        "rr_rmssd": ppg_feats["rr_rmssd"],
        "acdc_ratio": ppg_feats["acdc_ratio"],
        "ppg_sqi": ppg_feats["ppg_sqi"],
        "beat_count": ppg_feats["beat_count"],
        "acc_var": motion_feats["acc_var"],
        "motion_frac": motion_feats["motion_frac"],
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "hr_z": hr_z,
        "rr_rmssd_z": rr_z,
    }


def resolve_label(events, window_mid_ms):
    for evt in reversed(events):
        dt = ticks_diff(window_mid_ms, evt["ts"])
        if dt < 0:
            continue
        if dt <= LABEL_ACTIVE_MS:
            return evt["value"], evt["source"]
    return -1, "na"


def cleanup_events(events, cutoff_ms):
    while events and ticks_diff(cutoff_ms, events[0]["ts"]) > (LABEL_ACTIVE_MS * 2):
        events.pop(0)


def recompute_baseline(window_history, now_uptime_ms, config):
    lookback_ms = 5 * 60_000
    cutoff = now_uptime_ms - lookback_ms
    values_hr = []
    values_rr = []
    for w in window_history:
        if w["end_uptime_ms"] < cutoff:
            continue
        if w["dropped_by_qc"]:
            continue
        if w["label"] != 0:
            continue
        values_hr.append(w["hr_mean"])
        values_rr.append(w["rr_rmssd"])

    if not values_hr or not values_rr:
        print("[BASELINE] Not enough calm QC windows in last 5 min.")
        return False

    def mean_std(arr):
        mean = sum(arr) / len(arr)
        if len(arr) < 2:
            std = 1.0
        else:
            std = math.sqrt(sum((x - mean) ** 2 for x in arr) / (len(arr) - 1))
            if std < 1e-3:
                std = 1.0
        return mean, std

    hr_mean, hr_std = mean_std(values_hr)
    rr_mean, rr_std = mean_std(values_rr)

    baseline = config.setdefault("BASELINE", {})
    baseline["hr_mean"] = hr_mean
    baseline["hr_std"] = hr_std
    baseline["rr_rmssd"] = rr_mean
    baseline["rr_rmssd_std"] = rr_std
    save_config(config)
    print("[BASELINE] Updated baseline: hr_mean={:.1f}, hr_std={:.2f}, rr_rmssd={:.1f}, rr_std={:.2f}".format(
        hr_mean, hr_std, rr_mean, rr_std
    ))
    return True


# --------------------------------------------------------------------- main run
def main():
    boot_ticks = ticks_ms()
    config = load_config()

    print("[BOOT] Init I2C0 @400kHz")
    i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
    try:
        found = i2c.scan()
    except Exception:
        found = []
    print("[BOOT] I2C scan:", found)

    sensors = Sensors(i2c)
    print("[BOOT] Init MAX30102:", "OK" if sensors.has_ppg else "FAIL")
    if not sensors.has_ppg and sensors._last_ppg_error:
        print("[ERROR] MAX30102:", sensors._last_ppg_error)
    print("[BOOT] Init MPU6050:", "OK" if sensors.has_imu else "FAIL")
    if not sensors.has_imu and sensors._last_imu_error:
        print("[ERROR] MPU6050:", sensors._last_imu_error)

    dfplayer = init_dfplayer()
    print("[BOOT] Init DFPlayer:", "OK" if dfplayer else "FAIL")

    rf_model = RandomForestRuntime()
    rf_model.load()

    csv_logger = CSVLogger()
    log_path = csv_logger.current_path or "logs/unknown.csv"
    print("[SESSION] Logging to {}".format(log_path))

    button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_UP)
    last_button_state = button.value()
    last_button_change = ticks_ms()
    press_start = None

    if uselect and hasattr(uselect, "poll"):
        poller = uselect.poll()
        try:
            poller.register(sys.stdin, uselect.POLLIN)
        except Exception:
            poller = None
    else:
        poller = None

    label_events = []
    window_history = []
    last_prob = None
    last_alert_ms = - (config.get("MIN_ALERT_GAP_SEC", 120) * 1000)

    ppg_poll_due = ticks_ms()
    acc_poll_due = ticks_ms()
    next_window_due = ticks_add(ticks_ms(), WINDOW_STEP_MS)

    while True:
        now_ticks = ticks_ms()
        uptime_ms = ticks_diff(now_ticks, boot_ticks)

        # --------- PPG sampling
        if ticks_diff(now_ticks, ppg_poll_due) >= 0:
            # Drain as many samples as available this cycle.
            drained = False
            while True:
                sample = sensors.read_ppg_sample()
                if sample is None:
                    break
                sensors.feed_ppg(sample)
                drained = True
            ppg_poll_due = ticks_add(ppg_poll_due if drained else now_ticks, PPG_POLL_MS)

        # --------- Accel sampling
        if ticks_diff(now_ticks, acc_poll_due) >= 0:
            acc = sensors.read_acc_sample()
            if acc is not None:
                sensors.feed_acc(*acc)
            acc_poll_due = ticks_add(acc_poll_due, ACC_POLL_MS)

        # --------- Button handling (active-low)
        state = button.value()
        if state != last_button_state:
            if ticks_diff(now_ticks, last_button_change) >= BUTTON_DEBOUNCE_MS:
                last_button_change = now_ticks
                last_button_state = state
                if state == 0:
                    press_start = now_ticks
                else:
                    if press_start is not None:
                        duration = ticks_diff(now_ticks, press_start)
                        if duration < BUTTON_LONG_MS:
                            label_events.append({"ts": now_ticks, "value": 1, "source": "btn"})
                            print("[LABEL] Stressed (button short press)")
                        else:
                            label_events.append({"ts": now_ticks, "value": 0, "source": "btn"})
                            print("[LABEL] Calm (button long press)")
                        press_start = None
                        cleanup_events(label_events, now_ticks)

        # --------- Console commands
        if poller is not None:
            try:
                events = poller.poll(0)
            except Exception:
                events = None
                poller = None
            if events:
                try:
                    cmd = sys.stdin.read(1)
                except Exception:
                    cmd = None
                if cmd:
                    cmd = cmd.strip().lower()
                    if cmd == "s":
                        label_events.append({"ts": now_ticks, "value": 1, "source": "cmd"})
                        print("[LABEL] Simulated stressed")
                    elif cmd == "c":
                        label_events.append({"ts": now_ticks, "value": 0, "source": "cmd"})
                        print("[LABEL] Simulated calm")
                    elif cmd == "b":
                        recompute_baseline(window_history, uptime_ms, config)
                    elif cmd == "?":
                        print("[CONFIG]", config)
                        if last_prob is not None:
                            print("[STATE] Last prob={:.3f}".format(last_prob.get("prob", 0.0)))
                        else:
                            print("[STATE] No predictions yet.")
                    cleanup_events(label_events, now_ticks)

        # --------- Window processing
        if ticks_diff(now_ticks, next_window_due) >= 0:
            window_end_ticks = now_ticks
            window_start_ticks = ticks_add(window_end_ticks, -WINDOW_MS)
            window_mid_ticks = ticks_add(window_start_ticks, WINDOW_MS // 2)

            ppg_feats = sensors.get_ppg_features()
            motion_feats = sensors.get_motion_features(config.get("MOTION_THRESH_G", 0.15))

            rtc = rtc_available()
            if rtc:
                end_epoch = time.time()
                start_epoch = end_epoch - (WINDOW_MS / 1000.0)
                ts_start = isoformat_from_epoch(int(start_epoch))
                ts_end = isoformat_from_epoch(int(end_epoch))
                end_local = time.localtime()
                hour = end_local[3] + end_local[4] / 60.0
                angle = (hour / 24.0) * 2 * math.pi
                hour_sin = math.sin(angle)
                hour_cos = math.cos(angle)
            else:
                end_uptime = uptime_ms / 1000.0
                start_uptime = end_uptime - (WINDOW_MS / 1000.0)
                ts_start = isoformat_from_uptime(start_uptime)
                ts_end = isoformat_from_uptime(end_uptime)
                hour_sin, hour_cos = 0.0, 1.0

            baseline = config.get("BASELINE", {})
            hr_z = 0.0
            rr_z = 0.0
            if baseline.get("hr_mean") is not None and baseline.get("hr_std"):
                denom = baseline["hr_std"]
                hr_z = (ppg_feats["hr_mean"] - baseline["hr_mean"]) / denom if denom else 0.0
            if baseline.get("rr_rmssd") is not None and baseline.get("rr_rmssd_std"):
                denom = baseline["rr_rmssd_std"]
                rr_z = (ppg_feats["rr_rmssd"] - baseline["rr_rmssd"]) / denom if denom else 0.0

            feature_row = make_feature_vector(ppg_feats, motion_feats, hour_sin, hour_cos, hr_z, rr_z)

            label_value, label_src = resolve_label(label_events, window_mid_ticks)
            dropped_by_qc = 0
            if not sensors.has_ppg:
                dropped_by_qc = 1
            if ppg_feats["beat_count"] < 30 or ppg_feats["ppg_sqi"] < 0.8:
                dropped_by_qc = 1
            if motion_feats["motion_frac"] > 0.3:
                dropped_by_qc = 1
            if motion_feats["sample_count"] < 10:
                dropped_by_qc = 1

            stress_prob = ""
            alert_flag = ""
            prob_value = None
            if dropped_by_qc == 0:
                prob = rf_model.predict_proba(feature_row)
                if prob is not None:
                    prob_value = prob
                    stress_prob = "{:.3f}".format(prob)
                    last_prob = {"prob": prob, "ts": ts_end}
                    gap_ms = config.get("MIN_ALERT_GAP_SEC", 120) * 1000
                    if prob >= config.get("ALERT_THRESH", 0.6) and ticks_diff(now_ticks, last_alert_ms) >= gap_ms:
                        fired = play_alert(dfplayer)
                        if fired:
                            last_alert_ms = now_ticks
                        alert_flag = "1" if fired else "0"
                    else:
                        alert_flag = "0"
                else:
                    stress_prob = ""
                    alert_flag = ""
            else:
                prob_value = None
                alert_flag = ""

            row = [
                ts_start,
                ts_end,
                "{:.3f}".format(ppg_feats["hr_mean"]),
                "{:.3f}".format(ppg_feats["hr_std"]),
                "{:.3f}".format(ppg_feats["rr_rmssd"]),
                "{:.5f}".format(ppg_feats["acdc_ratio"]),
                "{:.3f}".format(ppg_feats["ppg_sqi"]),
                str(ppg_feats["beat_count"]),
                "{:.5f}".format(motion_feats["acc_var"]),
                "{:.3f}".format(motion_feats["motion_frac"]),
                "{:.5f}".format(hour_sin),
                "{:.5f}".format(hour_cos),
                "{:.3f}".format(hr_z),
                "{:.3f}".format(rr_z),
                str(label_value),
                label_src,
                str(dropped_by_qc),
                stress_prob,
                alert_flag,
            ]

            csv_logger.append(row)

            print("[WIN] ts={} HR={:.1f} RMSSD={:.1f} SQI={:.2f} MOTION={:.2f} QC={} label={} prob={} alert={}".format(
                ts_end,
                ppg_feats["hr_mean"],
                ppg_feats["rr_rmssd"],
                ppg_feats["ppg_sqi"],
                motion_feats["motion_frac"],
                dropped_by_qc,
                label_value,
                stress_prob or "NA",
                alert_flag or "0",
            ))

            window_history.append({
                "end_uptime_ms": uptime_ms,
                "hr_mean": ppg_feats["hr_mean"],
                "rr_rmssd": ppg_feats["rr_rmssd"],
                "dropped_by_qc": dropped_by_qc,
                "label": label_value,
            })
            if len(window_history) > 240:
                window_history.pop(0)

            next_window_due = ticks_add(next_window_due, WINDOW_STEP_MS)

        # Let other tasks run
        time.sleep_ms(5)


if __name__ == "__main__":  # pragma: no cover
    main()
