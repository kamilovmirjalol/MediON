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
PPG_MAX_BATCH_READS = 48  # upper bound used to drain FIFO faster when backlog builds up
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
MEDITATION_DURATION_OPTIONS = (
    (120000, "02:00"),
    (300000, "05:00"),
    (600000, "10:00"),
)

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

meditation_duration_ms = MEDITATION_DURATION_OPTIONS[1][0]
meditation_end_ms = 0
meditation_overlay = ""
meditation_overlay_sub = ""

# Display text tied directly to the track currently playing.
TRACK_DISPLAY_LINES = {
    1: ("Introduction",),
    2: ("Relax muscles",),
    3: ("Breathe in", "Pattern"),
    4: ("Breath normally",),
    5: ("Expand belly",),
    6: ("Relax muscles",),
    7: ("Relax muscles",),
    8: ("The end",),
}
TRACK_FORCE_FULL_PLAY = {8}
EXTRA_TRACKS = (5, 6, 7)

last_bpm_update_ms = 0
beat_led_until = 0
ppg = None
gc_pending_counter = 0

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
ML_UPDATE_INTERVAL_MS = 120
ML_MIN_BEATS = 2
ML_MIN_DURATION_S = 5.0
ML_MIN_SQI = 0.20
ml_feat_x = None
ml_last_tree_i = -1
ml_wait_last_print_ms = 0
ML_STRESS_CONFIRM_MS = 4000
ML_CALM_CONFIRM_MS = 2000
ml_display_state = 'unknown'
ml_display_prob = None
ml_pending_state = None
ml_pending_since = 0


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


def _randrange(limit):
    if limit <= 0:
        return 0
    try:
        val = os.urandom(1)[0]
    except Exception:
        val = ticks_ms() & 0xFF
    return val % limit


def _shuffle_extra_tracks():
    tracks = list(EXTRA_TRACKS)
    for idx in range(len(tracks) - 1, 0, -1):
        swap = _randrange(idx + 1)
        tracks[idx], tracks[swap] = tracks[swap], tracks[idx]
    return tracks


def set_meditation_overlay(text="", subtext=""):
    global meditation_overlay, meditation_overlay_sub
    meditation_overlay = text or ""
    meditation_overlay_sub = subtext or ""


def clear_meditation_overlay():
    set_meditation_overlay("", "")


def meditation_time_remaining_ms():
    if meditation_end_ms <= 0:
        return None
    now = ticks_ms()
    remain = ticks_diff(meditation_end_ms, now)
    if remain < 0:
        remain = 0
    return remain


def meditation_time_expired():
    remain = meditation_time_remaining_ms()
    return (remain is not None) and (remain == 0)


def select_meditation_duration():
    global meditation_duration_ms, meditation_end_ms
    meditation_end_ms = 0
    clear_meditation_overlay()
    # start from current selection if possible
    idx = 0
    for i, (ms, _) in enumerate(MEDITATION_DURATION_OPTIONS):
        if ms == meditation_duration_ms:
            idx = i
            break
    while True:
        oled.fill(0)
        oled.text("Meditation time", 0, 0)
        current = MEDITATION_DURATION_OPTIONS[idx]
        oled.text(current[1], 0, 20)
        oled.text("Short: cycle", 0, 40)
        oled.text("Hold: start", 0, 52)
        oled.show()
        if button.value() == 0:
            t0 = ticks_ms()
            while button.value() == 0:
                sleep(0.02)
            held = ticks_diff(ticks_ms(), t0)
            if held >= STRESS_HOLD_MS:
                meditation_duration_ms = current[0]
                return
            else:
                idx = (idx + 1) % len(MEDITATION_DURATION_OPTIONS)
        sleep(0.05)


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
    global MODE, display_waveform, record_start_ms, meditation_end_ms
    MODE = 'record'
    display_waveform = True
    meditation_end_ms = 0
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
    # Only write header when file is missing or empty; never rename or truncate.
    size = 0
    try:
        st = os.stat(CSV_FILENAME)
        size = st[6] if isinstance(st, (tuple, list)) and len(st) > 6 else (st[0] if isinstance(st, (tuple, list)) else 0)
    except OSError:
        size = 0
    if size and size > 0:
        return
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
    global current_hr, current_hrv, current_hr_smooth, last_bpm_update_ms, breath_baseline, breath_signal, breath_strength, breath_direction, last_gyro_value, wave_buf, beat_led_until, last_display_hr, last_display_hrv
    current_hr = None
    current_hr_smooth = None
    current_hrv = None
    last_bpm_update_ms = 0
    last_display_hr = None
    last_display_hrv = None
    breath_baseline = None
    breath_signal = 0.0
    breath_strength = 0.0
    breath_direction = "Stable"
    last_gyro_value = 0.0
    wave_buf = []
    beat_led_until = 0


# update_hr_state_from_ir removed; PPG handled by ppg.PPGProcessor
def pull_sensor_sample(max_reads=PPG_BATCH_READS):
    global last_debug_print_ms, last_red_value, last_ir_value, current_hr, current_hr_smooth, current_hrv, last_bpm_update_ms, wave_buf, last_display_hr, last_display_hrv, beat_led_until, ppg, gc_pending_counter
    red_val = None
    ir_val = None
    reads = 0
    while reads < max_reads:
        try:
            remaining = spo2.get_data_present()
        except Exception:
            remaining = 1 if reads == 0 else 0
        if remaining > 0 and (reads + remaining) > max_reads:
            max_reads = min(PPG_MAX_BATCH_READS, reads + remaining)
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
        finger_present = finger_on_sensor(ir)
        if not finger_present:
            reset_hr_processing()
            reads += 1
            continue
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
        gc_pending_counter += 1
        if gc_pending_counter >= 200:
            gc.collect()
            gc_pending_counter = 0
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
    """Return (decision, prob) using a smoothed ML output when available."""
    if ml_display_state in ('stress', 'calm') and ml_display_prob is not None:
        return (ml_display_state, ml_display_prob)
    if ml_prob is None or ml_is_stress is None:
        return ('unknown', None)
    return ('stress' if ml_is_stress else 'calm'), ml_prob


def _register_ml_output(prob):
    """Apply hysteresis so short stress spikes do not flip the UI immediately."""
    global ml_display_state, ml_display_prob, ml_pending_state, ml_pending_since
    if prob is None:
        return
    now = ticks_ms()
    raw_state = 'stress' if prob >= ml_threshold else 'calm'
    if ml_display_state not in ('stress', 'calm'):
        ml_display_state = raw_state
        ml_display_prob = prob
        ml_pending_state = None
        ml_pending_since = 0
        return
    if raw_state == ml_display_state:
        ml_display_prob = prob
        ml_pending_state = None
        ml_pending_since = 0
        return
    confirm_ms = ML_STRESS_CONFIRM_MS if raw_state == 'stress' else ML_CALM_CONFIRM_MS
    if confirm_ms <= 0:
        ml_display_state = raw_state
        ml_display_prob = prob
        ml_pending_state = None
        ml_pending_since = 0
        return
    if ml_pending_state != raw_state:
        ml_pending_state = raw_state
        ml_pending_since = now
        return
    if ml_pending_since and ticks_diff(now, ml_pending_since) >= confirm_ms:
        ml_display_state = raw_state
        ml_display_prob = prob
        ml_pending_state = None
        ml_pending_since = 0


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
                    # print every 8 trees to reduce console overhead
                    if (i is not None) and (n is not None) and (i != ml_last_tree_i) and (i % 8 == 0 or i == n):
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
                _register_ml_output(prob)
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


def _render_standard_display(header, detail, warning):
    hr_value = current_hr_smooth if current_hr_smooth is not None else current_hr
    if hr_value is None and last_display_hr is not None:
        hr_value = last_display_hr
    hr_str = "--" if hr_value is None else "{:3d}".format(hr_value)
    hrv_value = current_hrv if current_hrv is not None else last_display_hrv
    hrv_str = "--" if hrv_value is None else "{:3d}".format(hrv_value)
    line0 = "HR:{0} HRV:{1}".format(hr_str, hrv_str)
    oled.text(line0[:16], 0, 0)

    if MODE == 'meditate':
        label = "--"
        if ml_prob is not None and ml_is_stress is not None:
            conf = ml_prob if ml_is_stress else (1.0 - ml_prob)
            label = ("Stress" if ml_is_stress else "Calm") + " {0:.0f}%".format(conf*100)
        oled.text(("ML: " + label)[:16], 0, 12)

    show_wave = (MODE == 'record' and display_waveform)
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
                for i in range(min(WAVE_W, n)):
                    x = i
                    val = wave_buf[n - min(WAVE_W, n) + i]
                    y = WAVE_Y0 + (WAVE_H - 1) - int(((val - vmin) / vrng) * (WAVE_H - 1))
                    try:
                        oled.line(last_x, last_y, x, y, 1)
                    except Exception:
                        oled.pixel(x, y, 1)
                    last_x, last_y = x, y
        except Exception:
            pass
    else:
        if header:
            oled.text(header[:16], 0, 24)
            if len(header) > 16:
                oled.text(header[16:32][:16], 0, 36)
                if detail:
                    oled.text(detail[:16], 0, 48)
            elif detail:
                oled.text(detail[:16], 0, 36)
        elif detail:
            oled.text(detail[:16], 0, 24)

    now = ticks_ms()
    if beat_led_until and ticks_diff(beat_led_until, now) > 0:
        oled.fill_rect(124, 0, 4, 4, 1)

    if skip_notice and ticks_diff(skip_notice_expires, now) > 0:
        oled.text(skip_notice[:16], 0, 56)
    elif warning:
        oled.text(warning[:16], 0, 56)


def _render_meditation_display(warning):
    remain = meditation_time_remaining_ms()
    if remain is None:
        remain = 0
    mm = remain // 60000
    ss = (remain % 60000) // 1000
    lines = ["Time {:02d}:{:02d}".format(mm, ss)]
    decision, prob = ml_decision()
    if decision == 'stress':
        pct = int(min(max(prob or 0.0, 0.0), 1.0) * 100)
        lines.append("Stress {0:3d}%".format(pct))
    elif decision == 'calm':
        conf = 1.0 - (prob or 0.0)
        pct = int(min(max(conf, 0.0), 1.0) * 100)
        lines.append("Calm {0:3d}%".format(pct))
    else:
        lines.append("Preparing")
        lines.append("Stay relaxed")

    overlay_lines = []
    if meditation_overlay:
        overlay_lines.append(meditation_overlay)
        if meditation_overlay_sub:
            overlay_lines.append(meditation_overlay_sub)

    now = ticks_ms()
    notice = None
    if skip_notice and ticks_diff(skip_notice_expires, now) > 0:
        notice = skip_notice
    elif warning:
        notice = warning

    max_lines = 6
    reserve = 1 if notice else 0
    for extra in overlay_lines:
        if len(lines) >= (max_lines - reserve):
            break
        lines.append(extra)

    if notice:
        if len(lines) >= max_lines:
            lines[-1] = notice
        else:
            lines.append(notice)

    for idx, text in enumerate(lines[:max_lines]):
        if not text:
            continue
        oled.text(text[:16], 0, idx * 12)


def render_status(header="", detail="", warning=""):
    refresh_hr_timeout()
    oled.fill(0)
    if MODE == 'meditate' and meditation_end_ms > 0:
        _render_meditation_display(warning)
    else:
        _render_standard_display(header, detail, warning)
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
                # If event window has expired, finalize it immediately so a new event can start
                remain = RECORD_EVENT_WINDOW_MS - ticks_diff(now_ms, event_start_ms)
                if remain <= 0:
                    try:
                        finalize_current_event("auto")
                    except Exception:
                        pass
                else:
                    # show remaining time; do not start a new event yet
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


def play_and_monitor(track, header="", detail="", monitor_breath=False, allow_skip=True, respect_timer=True):
    movement_samples = 0
    movement_hits = 0
    overlay_text = None
    overlay_subtext = ""
    track_lines = TRACK_DISPLAY_LINES.get(track)
    if isinstance(track_lines, (tuple, list)):
        if track_lines:
            overlay_text = track_lines[0]
            if len(track_lines) > 1:
                overlay_subtext = track_lines[1]
    elif track_lines:
        overlay_text = str(track_lines)
    overlay_active = False
    if MODE == 'meditate' and overlay_text:
        set_meditation_overlay(overlay_text, overlay_subtext)
        overlay_active = True
    player.playRoot(track)
    time_expired = False
    timer_enforced = respect_timer and (track not in TRACK_FORCE_FULL_PLAY)
    while True:
        _, ir = pull_sensor_sample()
        if MODE == 'meditate':
            try:
                ml_update_nonblocking()
            except Exception:
                pass
            if timer_enforced and meditation_end_ms > 0 and ticks_diff(meditation_end_ms, ticks_ms()) <= 0:
                time_expired = True
                player.stop()
                break
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
        if short and allow_skip:
            player.stop()
            set_skip_notice("Track skipped")
            break
        busy = player.queryBusy()
        if busy is None:
            sleep(0.1)
        else:
            if not busy:
                break
        sleep(0.05)
    if overlay_active:
        clear_meditation_overlay()
    if time_expired:
        return 'done', movement_ratio(movement_hits, movement_samples)
    return 'done', movement_ratio(movement_hits, movement_samples)


def play_track(track, header, detail, monitor_breath=False):
    """Wrapper to ensure meditation tracks always run to completion."""
    return play_and_monitor(
        track,
        header,
        detail,
        monitor_breath=monitor_breath,
        allow_skip=False,
        respect_timer=False,
    )


def need_breath_focus():
    if MODE != 'meditate':
        return False
    decision, _ = ml_decision()
    return decision == 'stress'


def run_breath_focus_sequence():
    """Apply up to MAX_BREATH_RETRIES pattern cycles followed by track 0004."""
    attempt = 0
    while attempt < MAX_BREATH_RETRIES:
        attempt += 1
        header = "Breathing pattern"
        detail = "Track 0003 ({}/{})".format(attempt, MAX_BREATH_RETRIES)
        status, ratio = play_track(3, header, detail, monitor_breath=True)
        if status == 'stopped':
            return 'stopped'
        if ratio < MIN_BREATH_MOVEMENT_RATIO:
            status_move, _ = play_track(5, "Expand the belly", "Track 0005")
            if status_move == 'stopped':
                return 'stopped'
        if not need_breath_focus():
            status4, _ = play_track(4, "Breathe normally", "Track 0004")
            if status4 == 'stopped':
                return 'stopped'
            return 'calm'
    status4, _ = play_track(4, "Breathe normally", "Track 0004")
    if status4 == 'stopped':
        return 'stopped'
    return 'done'


def ensure_breath_focus_if_needed():
    """Check stress after an audio segment and prompt breathing focus once if needed."""
    if MODE != 'meditate':
        return 'ok'
    if not need_breath_focus():
        return 'ok'
    return run_breath_focus_sequence()




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


def end_session(message):
    global meditation_end_ms
    _ = message
    player.stop()
    clear_skip_notice()
    meditation_end_ms = 0
    clear_meditation_overlay()
    oled.fill(0)
    oled.text("Session ended.", 0, 0)
    oled.text("Thank you.", 0, 12)
    oled.show()
    sleep(1.5)


def run_session():
    global meditation_end_ms
    wait_for_ready()
    meditation_end_ms = ticks_ms() + meditation_duration_ms

    status, _ = play_track(1, "Introduction", "Track 0001")
    if status == 'stopped':
        end_session("Session stopped")
        return
    if ensure_breath_focus_if_needed() == 'stopped':
        end_session("Session stopped")
        return

    status, _ = play_track(2, "Relax muscles", "Track 0002")
    if status == 'stopped':
        end_session("Session stopped")
        return
    if ensure_breath_focus_if_needed() == 'stopped':
        end_session("Session stopped")
        return

    extras = _shuffle_extra_tracks()
    extra_headers = {
        5: "Expand the belly",
        6: "Relax your back",
        7: "Relax your face",
    }
    while True:
        if meditation_time_expired():
            break
        if not extras:
            extras = _shuffle_extra_tracks()
        track = extras.pop(0)
        header = extra_headers.get(track, "Guidance")
        detail = "Track {0:04d}".format(track)
        status, _ = play_track(track, header, detail)
        if status == 'stopped':
            end_session("Session stopped")
            return
        if ensure_breath_focus_if_needed() == 'stopped':
            end_session("Session stopped")
            return

    status, _ = play_track(8, "Closing meditation", "Track 0008")
    if status == 'stopped':
        end_session("Session stopped")
        return
    end_session("Session complete")


def main():
    while True:
        select = select_mode()
        if select == 'record':
            run_record_mode()
        else:
            select_meditation_duration()
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
