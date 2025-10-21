# main.py - MediON guided NSDR prototype

from machine import I2C, Pin
from utime import sleep, ticks_ms, ticks_diff
import gc
import uos as os

from ssd1306 import SSD1306_I2C
from dfplayer import DFPlayer
from max30102 import MAX30102
from mpu6050 import mpu6050
from ppg import PPGProcessor
try:
    import rf_infer as rfi
except Exception:
    rfi = None

# Hardware pins and comms
I2C_ID = 0
I2C_SDA = 0
I2C_SCL = 1
I2C_FREQ = 400000

UART_ID = 1
UART_TX = 4
UART_RX = 5
DFPLAYER_BUSY_PIN = 17

BUTTON_PIN = 15
HEADPHONE_PIN = 16
HEADPHONE_CONNECTED_LEVEL = 1  # change to 0 if your jack pulls the line LOW when connected

DEBUG = True
DEBUG_INTERVAL_MS = 600
# Session parameters
FINGER_IR_THRESHOLD = 8000
HR_CALM = 75
HRV_CALM_MS = 50
HR_END = 68
HRV_END_MS = 80
MAX_BREATH_RETRIES = 3
SESSION_MAX_CYCLES = 2
STRESS_HOLD_MS = 2000
SESSION_STOP_HOLD_MS = 5000
READY_HOLD_MS = 1500
PLAYER_VOLUME = 24

GYRO_AXIS = 'y'
GYRO_MOVEMENT_THRESHOLD = 3.0
MIN_BREATH_MOVEMENT_RATIO = 0.25
BREATH_AXIS = 'y'
BREATH_BASELINE_ALPHA = 0.05
BREATH_FILTER_ALPHA = 0.5
BREATH_STRENGTH_ALPHA = 0.2
BREATH_ACTIVITY_THRESHOLD = 0.035

IDLE_SAMPLE_DELAY = 0.12
PPG_BATCH_READS = 12
STALE_HR_TIMEOUT_MS = 8000
DISPLAY_STALE_TIMEOUT_MS = STALE_HR_TIMEOUT_MS * 2
# Legacy PPG detector constants above removed; using ppg.PPGProcessor

# CSV logging
CSV_LOG_ENABLED = True
CSV_FILENAME = "events.csv"
FEATURE_FIELDS = (
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn20",
    "sd1_ms",
    "sd2_ms",
    "amp_mean",
    "rise_ms_mean",
    "width50_ms_mean",
)
FEATURE_WINDOW_MS = 300000  # 5 minute rolling window
MODEL_WINDOW_MS = 15000     # short inference window (~15s) for instant ML updates
RECORD_EVENT_WINDOW_MS = 120000  # 2-minute per-event recording window

# Hardware setup
i2c = I2C(I2C_ID, sda=Pin(I2C_SDA), scl=Pin(I2C_SCL), freq=I2C_FREQ)
oled = SSD1306_I2C(128, 64, i2c)

player = DFPlayer(uartInstance=UART_ID, txPin=UART_TX, rxPin=UART_RX, busyPin=DFPLAYER_BUSY_PIN)
sleep(0.2)
player.setVolume(PLAYER_VOLUME)

spo2 = MAX30102(i2c_bus=I2C_ID, sda=I2C_SDA, scl=I2C_SCL)
sleep(0.2)

imu = mpu6050(i2c)

button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_UP)

try:
    headphone_pin = Pin(HEADPHONE_PIN, Pin.IN, Pin.PULL_UP)
except Exception:
    headphone_pin = None
 

# State variables
last_ir_value = 0
current_hr = None
current_hr_smooth = None
current_hrv = None

breath_direction = "Stable"
last_gyro_value = 0.0

skip_notice = ""
skip_notice_expires = 0
last_debug_print_ms = 0

breath_baseline = None
breath_signal = 0.0
breath_strength = 0.0
last_display_hr = None
last_display_hrv = None

last_bpm_update_ms = 0
beat_led_until = 0
ppg = None

# UI/mode control
MODE = None  # 'record' or 'meditate'
display_waveform = False
record_start_ms = 0
last_event_ms = 0  # legacy cooldown; not used for gating anymore
event_in_progress = False
event_label = None
event_start_ms = 0

# ML state
ml_head = None
ml_stream = None
ml_prob = None
ml_is_stress = None
ml_threshold = 0.5
ml_last_update_ms = 0
ml_next_update_ms = 0
ML_UPDATE_INTERVAL_MS = 800
ML_MIN_BEATS = 2
ML_MIN_DURATION_S = 5.0
ML_MIN_SQI = 0.20
ml_feat_x = None
ml_last_tree_i = -1
ml_wait_last_print_ms = 0


def init_ppg():
    global ppg
    ppg = PPGProcessor(fs=100)
    try:
        ppg.bfe.win_ms = FEATURE_WINDOW_MS
    except Exception:
        pass


init_ppg()


def init_ml():
    global ml_head, ml_stream, ml_threshold
    if rfi is None:
        return False
    mh = rfi.load_model_head("model_rf.json", log_fn=lambda *args, **kw: None)
    if not mh:
        return False
    ml_head = mh
    ml_threshold = float(mh.get("decision_threshold", mh.get("threshold", 0.5)))
    ml_stream = rfi.RFStreamer(ml_head)
    return True


init_ml()


def select_mode():
    global MODE, display_waveform, record_start_ms
    # UI prompt
    while True:
        oled.fill(0)
        oled.text("Select Mode", 0, 0)
        oled.text("Short: Record", 0, 16)
        oled.text("Hold: Meditate", 0, 32)
        oled.show()
        if button.value() == 0:
            t0 = ticks_ms()
            while button.value() == 0:
                sleep(0.02)
            held = ticks_diff(ticks_ms(), t0)
            if held >= STRESS_HOLD_MS:
                MODE = 'meditate'
                display_waveform = False
                return 'meditate'
            else:
                MODE = 'record'
                display_waveform = True
                record_start_ms = ticks_ms()
                return 'record'
        sleep(0.05)


def run_record_mode():
    global MODE, display_waveform, record_start_ms
    MODE = 'record'
    display_waveform = True
    # Minimal record UI loop; long-hold ends session
    while True:
        _, ir = pull_sensor_sample()
        finger_ok = finger_on_sensor(ir)
        warning = "" if finger_ok else "Finger?"
        # Render without header/detail to avoid overlap
        render_status("", "", warning)
        short, long_press = read_button_event()
        if long_press:
            # session summary on exit
            try:
                # finalize current event if in progress
                if event_in_progress:
                    finalize_current_event("manual_stop")
                # also write a session summary over the entire record duration
                ts = ticks_ms()
                win = ticks_diff(ts, record_start_ms)
                raw_features = ppg.bfe.features(ts, window_ms=win)
                feature_map = {
                    "mean_hr_bpm": raw_features.get("hr_mean_bpm", 0.0),
                    "sdnn_ms": raw_features.get("sdnn_ms", 0.0),
                    "rmssd_ms": raw_features.get("rmssd_ms", 0.0),
                    "pnn20": raw_features.get("pnn20", 0.0),
                    "sd1_ms": raw_features.get("sd1_ms", 0.0),
                    "sd2_ms": raw_features.get("sd2_ms", 0.0),
                    "amp_mean": raw_features.get("amp_mean", 0.0),
                    "rise_ms_mean": raw_features.get("rise_ms_mean", 0.0),
                    "width50_ms_mean": raw_features.get("width50_ms_mean", 0.0),
                }
                _log_event(ts_ms=ts, label="session_summary", feature_map=feature_map)
            except Exception:
                pass
            end_session("Session stopped")
            return
        sleep(0.05)

# Waveform ring buffer (filtered IR)
WAVE_W = 128
WAVE_H = 40
WAVE_Y0 = 24
wave_buf = []

# Last captured raw values for logging
last_red_value = None
last_ir_value = 0


def _ensure_csv_header():
    if not CSV_LOG_ENABLED:
        return
    try:
        st = os.stat(CSV_FILENAME)
        size = st[6] if isinstance(st, (tuple, list)) and len(st) > 6 else 0
        if size > 0:
            # verify header matches expected format
            try:
                with open(CSV_FILENAME, "r") as fh:
                    first = fh.readline().strip()
                expected = "timestamp_ms,label," + ",".join(FEATURE_FIELDS)
                if not first.startswith(expected):
                    # rename legacy file and create new header
                    try:
                        os.rename(CSV_FILENAME, "events_legacy.csv")
                    except Exception:
                        pass
                    size = 0
                else:
                    return
            except Exception:
                pass
    except OSError:
        pass
    try:
        with open(CSV_FILENAME, "a") as f:
            header = "timestamp_ms,label," + ",".join(FEATURE_FIELDS) + "\n"
            f.write(header)
    except OSError as e:
        if DEBUG:
            print("[CSV] header write failed:", e)


def _log_event(ts_ms, label, feature_map):
    if not CSV_LOG_ENABLED:
        return
    try:
        with open(CSV_FILENAME, "a") as f:
            row = [ts_ms, label]
            for key in FEATURE_FIELDS:
                val = feature_map.get(key) if feature_map else None
                if val is None:
                    row.append("")
                else:
                    row.append("{:.4f}".format(val) if isinstance(val, (int, float)) else str(val))
            f.write(",".join(str(x) for x in row) + "\n")
    except OSError as e:
        if DEBUG:
            print("[CSV] append failed:", e)

# Prepare CSV header if needed (create file and header once)
_ensure_csv_header()


def clear_skip_notice():
    global skip_notice, skip_notice_expires
    skip_notice = ""
    skip_notice_expires = 0


def set_skip_notice(text, duration_ms=2500):
    global skip_notice, skip_notice_expires
    skip_notice = text
    skip_notice_expires = ticks_ms() + duration_ms


def headphones_connected():
    if headphone_pin is None:
        return True
    level = headphone_pin.value()
    return level == HEADPHONE_CONNECTED_LEVEL


def refresh_hr_timeout():
    global current_hr, current_hrv, current_hr_smooth, last_display_hr, last_display_hrv
    if last_bpm_update_ms == 0:
        return
    now = ticks_ms()
    age = ticks_diff(now, last_bpm_update_ms)
    if age > STALE_HR_TIMEOUT_MS:
        current_hr = None
        current_hr_smooth = None
        if age > DISPLAY_STALE_TIMEOUT_MS:
            last_display_hr = None
            last_display_hrv = None


def reset_hr_processing():
    global current_hr, current_hrv, current_hr_smooth, last_bpm_update_ms, breath_baseline, breath_signal, breath_strength, breath_direction, last_gyro_value, wave_buf, beat_led_until
    current_hr = None
    current_hr_smooth = None
    current_hrv = None
    last_bpm_update_ms = 0
    breath_baseline = None
    breath_signal = 0.0
    breath_strength = 0.0
    breath_direction = "Stable"
    last_gyro_value = 0.0
    wave_buf = []
    beat_led_until = 0


# update_hr_state_from_ir removed; PPG handled by ppg.PPGProcessor
def pull_sensor_sample(max_reads=PPG_BATCH_READS):
    global last_debug_print_ms, last_red_value, last_ir_value, current_hr, current_hr_smooth, current_hrv, last_bpm_update_ms, wave_buf, last_display_hr, last_display_hrv, beat_led_until, ppg
    red_val = None
    ir_val = None
    reads = 0
    while reads < max_reads:
        try:
            remaining = spo2.get_data_present()
        except Exception:
            remaining = 1 if reads == 0 else 0
        if remaining <= 0:
            if reads == 0:
                remaining = 1
            else:
                break
        try:
            red, ir = spo2.read_fifo()
        except Exception:
            break
        now = ticks_ms()
        last_red_value = red
        last_ir_value = ir
        # Acc vertical in g
        try:
            accel = imu.get_accel_data(g=True)
            acc_vert_g = accel.get(BREATH_AXIS, 0.0)
        except Exception:
            acc_vert_g = 0.0
        try:
            out = ppg.process_sample(now, ir, acc_vert_g, 0.0, 0.0)
        except MemoryError:
            if DEBUG:
                print("[PPG] memory error, resetting pipeline")
            gc.collect()
            init_ppg()
            reset_hr_processing()
            gc.collect()
            continue
        # Update HR/HRV from PPG processor
        if out:
            bpm = out.get('bpm_avg')
            hrv = out.get('hrv')
            blink_until = out.get('beat_led_until')
            if blink_until is not None:
                beat_led_until = blink_until
            if bpm:
                current_hr = int(bpm)
                current_hr_smooth = current_hr
                last_bpm_update_ms = now
                last_display_hr = current_hr
            if hrv is not None:
                try:
                    current_hrv = int(hrv)
                    last_display_hrv = current_hrv
                except Exception:
                    current_hrv = None
                
            # Append filtered waveform
            ir_filt = out.get('ir_filtered')
            if ir_filt is not None:
                wave_buf.append(ir_filt)
                if len(wave_buf) > WAVE_W:
                    del wave_buf[:len(wave_buf) - WAVE_W]
        red_val, ir_val = red, ir
        reads += 1
        if remaining <= 1:
            break
    if DEBUG:
        now = ticks_ms()
        if last_debug_print_ms == 0 or ticks_diff(now, last_debug_print_ms) >= DEBUG_INTERVAL_MS:
            hp_level = headphone_pin.value() if headphone_pin else "n/a"
            hp_connected = headphones_connected()
            hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
            hr_display = hr_value if hr_value is not None else "--"
            hrv_value = current_hrv if current_hrv is not None else last_display_hrv
            hrv_display = hrv_value if hrv_value is not None else "--"
            finger_state = finger_on_sensor(ir_val)
            print("[DBG] IR:{} finger:{} HR:{} HRV:{} HP_level:{} HP_connected:{}".format(
                  ir_val if ir_val is not None else "None",
                  finger_state,
                  hr_display,
                  hrv_display,
                  hp_level,
                  hp_connected))
            last_debug_print_ms = now
    if reads:
        gc.collect()
    return red_val, ir_val


def finger_on_sensor(ir_value):
    if ir_value is None:
        return False
    return ir_value >= FINGER_IR_THRESHOLD



def capture_event(label):
    try:
        ts = ticks_ms()
        hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
        hrv_value = current_hrv if current_hrv is not None else last_display_hrv
        try:
            raw_features = ppg.bfe.features(ts, window_ms=FEATURE_WINDOW_MS)  # 5-minute window
        except Exception:
            raw_features = None

        if raw_features:
            feature_map = {
                "mean_hr_bpm": raw_features.get("hr_mean_bpm") or (hr_value if hr_value is not None else 0.0),
                "sdnn_ms": raw_features.get("sdnn_ms"),
                "rmssd_ms": raw_features.get("rmssd_ms"),
                "pnn20": raw_features.get("pnn20"),
                "sd1_ms": raw_features.get("sd1_ms"),
                "sd2_ms": raw_features.get("sd2_ms"),
                "amp_mean": raw_features.get("amp_mean"),
                "rise_ms_mean": raw_features.get("rise_ms_mean"),
                "width50_ms_mean": raw_features.get("width50_ms_mean"),
            }
        else:
            feature_map = {
                "mean_hr_bpm": hr_value if hr_value is not None else "",
                "sdnn_ms": hrv_value if hrv_value is not None else "",
                "rmssd_ms": "",
                "pnn20": "",
                "sd1_ms": "",
                "sd2_ms": "",
                "amp_mean": "",
                "rise_ms_mean": "",
                "width50_ms_mean": "",
            }

        _log_event(ts_ms=ts, label=label, feature_map=feature_map)

        # Update last event timestamp for cooldown between labels
        global last_event_ms
        last_event_ms = ts

        # Console output for live monitoring
        try:
            summary_parts = []
            for name in FEATURE_FIELDS:
                val = feature_map.get(name)
                if isinstance(val, (int, float)) and val is not None:
                    summary_parts.append("{}={:.2f}".format(name, val))
                elif val not in (None, ""):
                    summary_parts.append("{}={}".format(name, val))
                else:
                    summary_parts.append("{}=".format(name))
            print("[FEATURES] {} {}".format(label.upper(), " ".join(summary_parts)))
        except Exception:
            pass
    except Exception as e:
        if DEBUG:
            print("[EVT] capture failed:", e)


def detect_breath_motion():
    global breath_direction, last_gyro_value, breath_baseline, breath_signal, breath_strength
    try:
        accel = imu.get_accel_data(g=True)
        value = accel.get(BREATH_AXIS, 0.0)
    except Exception:
        value = 0.0
    if breath_baseline is None:
        breath_baseline = value
    breath_baseline += BREATH_BASELINE_ALPHA * (value - breath_baseline)
    delta = value - breath_baseline
    breath_signal += BREATH_FILTER_ALPHA * (delta - breath_signal)
    amplitude = abs(breath_signal)
    last_gyro_value = amplitude
    breath_strength += BREATH_STRENGTH_ALPHA * (amplitude - breath_strength)
    moved = amplitude >= BREATH_ACTIVITY_THRESHOLD
    if breath_signal > BREATH_ACTIVITY_THRESHOLD:
        breath_direction = "Expand"
    elif breath_signal < -BREATH_ACTIVITY_THRESHOLD:
        breath_direction = "Lower"
    else:
        breath_direction = "Stable"
    return moved, breath_direction, breath_strength


def compose_warning_line(finger_ok, headphone_ok):
    warnings = []
    if not finger_ok:
        warnings.append("Finger?")
    if not headphone_ok:
        warnings.append("Plug HP")
    return " ".join(warnings)


def ml_decision():
    """Return (decision, prob) using the last computed ML output if available.
    Does not run inference; call ml_update_nonblocking() elsewhere to progress.
    """
    if ml_prob is None or ml_is_stress is None:
        return ('unknown', None)
    return ('stress' if ml_is_stress else 'calm'), ml_prob


def ml_update_nonblocking():
    """Incremental ML update to keep meditation mode responsive.
    - Recomputes features at most once per ML_UPDATE_INTERVAL_MS
    - Streams a few trees per call to avoid blocking
    - Updates global ml_prob/ml_is_stress when done
    """
    global ml_feat_x, ml_next_update_ms, ml_prob, ml_is_stress, ml_last_update_ms, ml_last_tree_i, ml_wait_last_print_ms
    if not (ml_head and ml_stream):
        return
    now = ticks_ms()
    # If no job in flight and it's time, start a new evaluation
    if ml_stream.done() and (ml_next_update_ms == 0 or ticks_diff(now, ml_next_update_ms) >= 0):
        # compute features
        try:
            feats = ppg.bfe.features(now, window_ms=MODEL_WINDOW_MS)
        except Exception:
            feats = None
        data_ok = False
        if feats:
            n_beats = int(feats.get('n_beats', 0) or 0)
            duration_s = float(feats.get('event_duration_s', 0.0) or 0.0)
            sqi = float(feats.get('sqi', 0.0) or 0.0)
            data_ok = (n_beats >= ML_MIN_BEATS) and (duration_s >= ML_MIN_DURATION_S) and (sqi >= ML_MIN_SQI)
            if DEBUG and not data_ok:
                # Throttle waiting message to ~5s
                if (ml_wait_last_print_ms == 0) or (ticks_diff(now, ml_wait_last_print_ms) >= 5000):
                    print("[ML] waiting data beats:{} dur:{:.0f}s sqi:{:.2f}".format(n_beats, duration_s, sqi))
                    ml_wait_last_print_ms = now
        if data_ok:
            fmap = {
                'mean_hr_bpm': feats.get('hr_mean_bpm', 0.0),
                'sdnn_ms': feats.get('sdnn_ms', 0.0),
                'rmssd_ms': feats.get('rmssd_ms', 0.0),
                'pnn20': feats.get('pnn20', 0.0),
                'sd1_ms': feats.get('sd1_ms', 0.0),
                'sd2_ms': feats.get('sd2_ms', 0.0),
                'amp_mean': feats.get('amp_mean', 0.0),
                'rise_ms_mean': feats.get('rise_ms_mean', 0.0),
                'width50_ms_mean': feats.get('width50_ms_mean', 0.0),
            }
            try:
                ml_feat_x = rfi.make_feature_vector(ml_head, fmap)
                ml_stream.reset()
                ml_stream.start(ml_feat_x)
                ml_last_tree_i = -1
            except Exception:
                ml_feat_x = None
        # schedule next feature refresh
        ml_next_update_ms = now + ML_UPDATE_INTERVAL_MS
    # Step a few trees if evaluating
    if not ml_stream.done():
        try:
            ml_stream.step(1)
            # Console progress of trees processed
            if DEBUG:
                try:
                    i = getattr(ml_stream, "_i", None)
                    n = getattr(ml_stream, "n", None)
                    # print every 4 trees to reduce console overhead
                    if (i is not None) and (n is not None) and (i != ml_last_tree_i) and (i % 4 == 0 or i == n):
                        print("[ML] trees {}/{}".format(i, n))
                        ml_last_tree_i = i
                except Exception:
                    pass
        except Exception:
            pass
        if ml_stream.done():
            try:
                prob = ml_stream.prob()
                ml_prob = prob
                ml_is_stress = (prob >= ml_threshold)
                ml_last_update_ms = now
                if DEBUG:
                    print("[ML] done {} prob={:.3f}".format(getattr(ml_stream, "n", 0), prob))
            except Exception:
                pass


# -------- Recording helpers (2-minute event windows) --------
def start_record_event(label):
    global event_in_progress, event_label, event_start_ms
    if event_in_progress:
        return False
    event_in_progress = True
    event_label = label
    event_start_ms = ticks_ms()
    # Show 2:00 countdown initially
    set_skip_notice("REC %s 02:00" % label.upper(), 1200)
    return True


def finalize_current_event(reason="auto"):
    global event_in_progress, event_label, event_start_ms, last_event_ms
    if not event_in_progress or event_label is None:
        return False
    ts = ticks_ms()
    elapsed = ticks_diff(ts, event_start_ms)
    if elapsed < 0:
        elapsed = 0
    try:
        raw = ppg.bfe.features(ts, window_ms=elapsed if elapsed > 0 else 1)
    except Exception:
        raw = None
    # Build feature map (fallback to zeros if no raw)
    fmap = {
        "mean_hr_bpm": (raw.get("hr_mean_bpm") if raw else 0.0),
        "sdnn_ms": (raw.get("sdnn_ms") if raw else 0.0),
        "rmssd_ms": (raw.get("rmssd_ms") if raw else 0.0),
        "pnn20": (raw.get("pnn20") if raw else 0.0),
        "sd1_ms": (raw.get("sd1_ms") if raw else 0.0),
        "sd2_ms": (raw.get("sd2_ms") if raw else 0.0),
        "amp_mean": (raw.get("amp_mean") if raw else 0.0),
        "rise_ms_mean": (raw.get("rise_ms_mean") if raw else 0.0),
        "width50_ms_mean": (raw.get("width50_ms_mean") if raw else 0.0),
    }
    try:
        _log_event(ts_ms=ts, label=event_label, feature_map=fmap)
    except Exception as e:
        if DEBUG:
            print("[REC] log failed:", e)
    # Mark finished
    last_event_ms = ts
    set_skip_notice("Saved %s" % event_label.upper(), 1200)
    event_in_progress = False
    event_label = None
    event_start_ms = 0
    return True


def tick_record_event_ui():
    # Called frequently in loops to update countdown and auto-finalize
    if MODE != 'record' or not event_in_progress:
        return
    now = ticks_ms()
    remain = RECORD_EVENT_WINDOW_MS - ticks_diff(now, event_start_ms)
    if remain <= 0:
        finalize_current_event("auto")
        return
    # Update countdown display briefly
    mm = remain // 60000
    ss = (remain % 60000) // 1000
    set_skip_notice("Left %dm%02ds" % (mm, ss), 600)


def render_status(header="", detail="", warning=""):
    refresh_hr_timeout()
    oled.fill(0)
    hr_value = current_hr_smooth if current_hr_smooth is not None else current_hr
    if hr_value is None and last_display_hr is not None:
        hr_value = last_display_hr
    hr_str = "--" if hr_value is None else "{:3d}".format(hr_value)
    hrv_value = current_hrv if current_hrv is not None else last_display_hrv
    hrv_str = "--" if hrv_value is None else "{:3d}".format(hrv_value)
    line0 = "HR:{0} HRV:{1}".format(hr_str, hrv_str)
    oled.text(line0[:16], 0, 0)

    if MODE == 'meditate':
        # Show ML status instead of breath. Display confidence for the predicted class.
        label = "--"
        if ml_prob is not None and ml_is_stress is not None:
            conf = ml_prob if ml_is_stress else (1.0 - ml_prob)
            label = ("Stress" if ml_is_stress else "Calm") + " {0:.0f}%".format(conf*100)
        oled.text(("ML: " + label)[:16], 0, 12)

    # Draw waveform area (auto-scaled)
    # Also show in meditation to visualize responsiveness for debugging
    show_wave = (MODE == 'record' and display_waveform) or (MODE == 'meditate')
    if show_wave:
        try:
            oled.fill_rect(0, WAVE_Y0, 128, WAVE_H, 0)
            if wave_buf:
                vmin = min(wave_buf)
                vmax = max(wave_buf)
                vrng = (vmax - vmin) if (vmax != vmin) else 1.0
                last_x = 0
                last_y = WAVE_Y0 + (WAVE_H // 2)
                n = len(wave_buf)
                # Map up to 128 samples to screen width
                for i in range(min(WAVE_W, n)):
                    x = i
                    val = wave_buf[n - min(WAVE_W, n) + i]
                    y = WAVE_Y0 + (WAVE_H - 1) - int(((val - vmin) / vrng) * (WAVE_H - 1))
                    # draw line from last to current
                    try:
                        oled.line(last_x, last_y, x, y, 1)
                    except Exception:
                        # fallback: set pixel
                        oled.pixel(x, y, 1)
                    last_x, last_y = x, y
        except Exception:
            pass

    # Only draw header/detail when not showing waveform
    if (not show_wave) and header:
        oled.text(header[:16], 0, 24)
        if len(header) > 16:
            oled.text(header[16:32][:16], 0, 36)
            if detail:
                oled.text(detail[:16], 0, 48)
        elif detail:
            oled.text(detail[:16], 0, 36)
    elif (not show_wave) and detail:
        oled.text(detail[:16], 0, 24)

    now = ticks_ms()
    if beat_led_until and ticks_diff(beat_led_until, now) > 0:
        oled.fill_rect(124, 0, 4, 4, 1)

    if skip_notice and ticks_diff(skip_notice_expires, now) > 0:
        oled.text(skip_notice[:16], 0, 56)
    elif warning:
        oled.text(warning[:16], 0, 56)
    oled.show()


def read_button_event():
    if button.value() == 0:
        start = ticks_ms()
        while button.value() == 0:
            hold = ticks_diff(ticks_ms(), start)
            if hold >= SESSION_STOP_HOLD_MS:
                while button.value() == 0:
                    sleep(0.02)
                return False, True
            sleep(0.02)
        hold_duration = ticks_diff(ticks_ms(), start)
        if MODE == 'record':
            now_ms = ticks_ms()
            if event_in_progress:
                # show remaining time; do not start a new event
                remain = RECORD_EVENT_WINDOW_MS - ticks_diff(now_ms, event_start_ms)
                if remain < 0: remain = 0
                set_skip_notice("Left %dm%02ds" % (remain//60000, (remain%60000)//1000), 1500)
                return True, False
            # Start an event depending on press length
            if hold_duration >= STRESS_HOLD_MS:
                if start_record_event("stress"):
                    set_skip_notice("REC STRESS 02:00", 1200)
                return False, False
            else:
                if start_record_event("calm"):
                    set_skip_notice("REC CALM 02:00", 1200)
                return True, False
        else:
            # meditation mode: no event logging on presses (only long-hold handled above)
            return True, False
    return False, False


def movement_ratio(hits, samples):
    if samples <= 0:
        return 0.0
    return hits / samples


def monitor_idle(duration_ms, header="", detail=""):
    start = ticks_ms()
    while ticks_diff(ticks_ms(), start) < duration_ms:
        _, ir = pull_sensor_sample()
        # opportunistically update ML once per idle loop
        if MODE == 'meditate':
            try:
                ml_update_nonblocking()
            except Exception:
                pass
        if MODE == 'record':
            tick_record_event_ui()
        detect_breath_motion()
        finger_ok = finger_on_sensor(ir)
        headphone_ok = headphones_connected()
        warning = compose_warning_line(finger_ok, headphone_ok)
        render_status(header, detail, warning)
        short, long_press = read_button_event()
        if long_press:
            set_skip_notice("Session stopped")
            return 'stop'
        sleep(IDLE_SAMPLE_DELAY)
    return 'ok'


def play_and_monitor(track, header="", detail="", monitor_breath=False, allow_skip=True):
    movement_samples = 0
    movement_hits = 0
    player.playRoot(track)
    while True:
        _, ir = pull_sensor_sample()
        if MODE == 'meditate':
            try:
                ml_update_nonblocking()
            except Exception:
                pass
        elif MODE == 'record':
            tick_record_event_ui()
        moved, _, _ = detect_breath_motion()
        finger_ok = finger_on_sensor(ir)
        headphone_ok = headphones_connected()
        if monitor_breath:
            movement_samples += 1
            if moved:
                movement_hits += 1
        warning = compose_warning_line(finger_ok, headphone_ok)
        render_status(header, detail, warning)
        short, long_press = read_button_event()
        if long_press:
            player.stop()
            set_skip_notice("Session stopped")
            return 'stopped', movement_ratio(movement_hits, movement_samples)
        busy = player.queryBusy()
        if busy is None:
            sleep(0.1)
        else:
            if not busy:
                break
        sleep(0.05)
    return 'done', movement_ratio(movement_hits, movement_samples)


def ensure_intro_track():
    while True:
        status, _ = play_and_monitor(1, "Introduction", "Track 0001", monitor_breath=False, allow_skip=False)
        if status == 'stopped':
            return 'stopped'
        if status == 'done':
            return 'done'


def metrics_calm():
    # ML primary (2-min window); heuristics fallback
    decision, _ = ml_decision()
    if decision == 'calm':
        return True
    if decision == 'stress':
        return False
    # Fallback heuristics
    hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
    if hr_value is None:
        return False
    if hr_value > 100:
        return False
    if 50 <= hr_value <= 100:
        return True
    # hr < 50 => treat as calm conservatively
    return True


def end_condition_met():
    decision, _ = ml_decision()
    if decision == 'calm':
        return True
    if decision == 'stress':
        return False
    # fallback heuristics
    hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
    if hr_value is None:
        return False
    if hr_value > 100:
        return False
    if 50 <= hr_value <= 100:
        return True
    return True


def handle_breath_retries():
    attempt = 0
    while attempt < MAX_BREATH_RETRIES:
        attempt += 1
        header = "Breathing pattern"
        detail = "Track 0003 (%d/%d)" % (attempt, MAX_BREATH_RETRIES)
        status, ratio = play_and_monitor(3, header, detail, monitor_breath=True)
        if status == 'stopped':
            return 'stopped'
        if ratio < MIN_BREATH_MOVEMENT_RATIO:
            set_skip_notice("Expand your belly")
            status_move, _ = play_and_monitor(5, "Expand the belly", "Track 0005")
            if status_move == 'stopped':
                return 'stopped'
        result = monitor_idle(2000, "Checking calm", "Hold steady")
        if result == 'stop':
            return 'stopped'
        if metrics_calm():
            confirm_status, _ = play_and_monitor(4, "Breathe normally", "Track 0004")
            if confirm_status == 'stopped':
                return 'stopped'
            return 'calm'
    return 'not_calm'


def muscle_focus_stage():
    status_back, _ = play_and_monitor(6, "Relax your back", "Track 0006")
    if status_back == 'stopped':
        return 'stopped'
    status_face, _ = play_and_monitor(7, "Relax your face", "Track 0007")
    if status_face == 'stopped':
        return 'stopped'
    return 'done'


def wait_for_ready():
    clear_skip_notice()
    hold_since = None
    while True:
        _, ir = pull_sensor_sample()
        detect_breath_motion()
        finger_ok = finger_on_sensor(ir)
        headphone_ok = headphones_connected()
        if finger_ok and headphone_ok:
            if hold_since is None:
                hold_since = ticks_ms()
            header = "All sensors ready"
            detail = "Starting shortly"
            if ticks_diff(ticks_ms(), hold_since) >= READY_HOLD_MS:
                return
        else:
            hold_since = None
            if not finger_ok and not headphone_ok:
                header = "Place finger &"
                detail = "connect headphones"
            elif not finger_ok:
                header = "Place finger on"
                detail = "the sensor"
            else:
                header = "Plug headphones"
                detail = "to begin"
        warning = compose_warning_line(finger_ok, headphone_ok)
        render_status("Welcome MediON", header, warning)
        short, long_press = read_button_event()
        if long_press:
            set_skip_notice("Hold released")
        elif short and finger_ok and headphone_ok:
            return
        sleep(0.15)


def final_relaxation_check():
    result = monitor_idle(2000, "Settling", "Evaluating calm")
    if result == 'stop':
        return 'stopped'
    if end_condition_met() or metrics_calm():
        status, _ = play_and_monitor(8, "Closing meditation", "Track 0008", allow_skip=False)
        if status == 'stopped':
            return 'stopped'
        return 'done'
    return 'continue'


def end_session(message):
    player.stop()
    clear_skip_notice()
    render_status(message, "Thank you", "")
    sleep(1.5)
    oled.fill(0)
    oled.text("MediON ready", 0, 0)
    oled.show()


def run_session():
    wait_for_ready()
    status = ensure_intro_track()
    if status == 'stopped':
        end_session("Session stopped")
        return
    final_message = "Session ended"
    cycle = 0
    while cycle < SESSION_MAX_CYCLES:
        cycle += 1
        result = monitor_idle(1500, "Collecting HR", "Preparing next")
        if result == 'stop':
            final_message = "Session stopped"
            break
        if metrics_calm():
            status, _ = play_and_monitor(2, "Body sensations", "Track 0002")
            if status == 'stopped':
                final_message = "Session stopped"
                break
        else:
            breath_result = handle_breath_retries()
            if breath_result == 'stopped':
                final_message = "Session stopped"
                break
            if breath_result == 'not_calm':
                set_skip_notice("Continuing guidance")
        muscle_result = muscle_focus_stage()
        if muscle_result == 'stopped':
            final_message = "Session stopped"
            break
        end_status = final_relaxation_check()
        if end_status == 'stopped':
            final_message = "Session stopped"
            break
        if end_status == 'done':
            final_message = "Session complete"
            break
        render_status("Continuing support", "Another cycle", "")
        sleep(0.8)
    end_session(final_message)


def main():
    select = select_mode()
    if select == 'record':
        run_record_mode()
    else:
        run_session()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # On abrupt exit, write session summary if recording
        try:
            if MODE == 'record':
                ts = ticks_ms()
                win = ticks_diff(ts, record_start_ms) if record_start_ms else FEATURE_WINDOW_MS
                raw_features = ppg.bfe.features(ts, window_ms=win)
                feature_map = {
                    "mean_hr_bpm": raw_features.get("hr_mean_bpm", 0.0),
                    "sdnn_ms": raw_features.get("sdnn_ms", 0.0),
                    "rmssd_ms": raw_features.get("rmssd_ms", 0.0),
                    "pnn20": raw_features.get("pnn20", 0.0),
                    "sd1_ms": raw_features.get("sd1_ms", 0.0),
                    "sd2_ms": raw_features.get("sd2_ms", 0.0),
                    "amp_mean": raw_features.get("amp_mean", 0.0),
                    "rise_ms_mean": raw_features.get("rise_ms_mean", 0.0),
                    "width50_ms_mean": raw_features.get("width50_ms_mean", 0.0),
                }
                _log_event(ts_ms=ts, label="session_summary", feature_map=feature_map)
        except Exception:
            pass
        end_session("Session stopped")
