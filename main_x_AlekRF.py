# main.py - final-final MediON prototype runner
# Assumes these files are present on the Pico:
# dfplayer.py, max30102.py, mpu6050.py (MicroPython version), ssd1306.py

from machine import Pin, I2C
from utime import sleep, ticks_ms, ticks_diff
import math
import json
try:
    import uos as os
except ImportError:
    import os

# --- Imports for drivers (your existing files) ---
from ssd1306 import SSD1306_I2C
from dfplayer import DFPlayer
from max30102 import MAX30102
from mpu6050 import mpu6050

# -----------------------------
# Config & thresholds (tweak these)
# -----------------------------
FINGER_IR_THRESHOLD = 50000     # IR value above this means finger is on sensor
HR_CALM = 75                    # bpm threshold to consider relaxed
HRV_CALM_MS = 50                # ms threshold for HRV to consider relaxed
HR_END = 70                     # bpm threshold to end session
HRV_END_MS = 80                 # HRV threshold to end session
MAX_RETRIES = 3                 # number of breathing retries (0003)
BREATH_MOVEMENT_THRESHOLD = 0.02  # g change on chosen accel axis to detect breathing movement
LONG_PRESS_MS = 2000            # long press duration to stop session (ms)
GRAPH_SCALE = 100000            # scale factor for drawing graph (tweak if necessary)

MODEL_HEADER_PATH = 'model_rf.json'
MODEL_TREE_DIR = 'trees_rf'
MODEL_WINDOW_MS = 60000
MODEL_MIN_BEATS = 4
MODEL_MAX_PROB_AGE_MS = 15000

# -----------------------------
# Hardware setup
# -----------------------------
# I2C bus (your wiring: GP0=SDA, GP1=SCL)
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)

class _NullOLED:
    def fill(self, *args):
        pass
    def fill_rect(self, *args):
        pass
    def text(self, *args):
        pass
    def show(self):
        pass

try:
    oled = SSD1306_I2C(128, 64, i2c)
    display_ready = True
except Exception as exc:
    print('OLED init failed:', exc)
    oled = _NullOLED()
    display_ready = False

# DFPlayer (UART1, GP4=TX (to module RX), GP5=RX (from module TX), BUSY=GP17)
player = DFPlayer(uartInstance=1, txPin=4, rxPin=5, busyPin=17)
sleep(0.2)  # give DFPlayer a moment
player.setVolume(25)

# MAX30102 (your driver expects i2c_bus=0, sda=0, scl=1)
max30102 = MAX30102(i2c_bus=0, sda=0, scl=1)
sleep(0.2)

# MPU6050 (micropython mpu6050 class)
mpu = mpu6050(i2c)

# Button for start/skip/stop (GP15). Assumes PULL_UP: idle HIGH, pressed LOW
btn_pin = Pin(15, Pin.IN, Pin.PULL_UP)

# -----------------------------
# Helper utilities
# -----------------------------
def oled_clear():
    oled.fill(0)
    oled.show()

def show_centered(line1="", line2="", line3=""):
    oled.fill(0)
    if line1: oled.text(line1, 0, 0)
    if line2: oled.text(line2, 0, 12)
    if line3: oled.text(line3, 0, 24)
    oled.show()

# Read finger presence by checking IR reading
def finger_present():
    # try to read one sample safely
    try:
        # if FIFO empty, read_fifo may raise; handle gracefully
        red, ir = max30102.read_fifo()
        return ir >= FINGER_IR_THRESHOLD, red, ir
    except Exception as e:
        return False, 0, 0

# Heart detection (simple): rising-edge threshold
last_ir = 0
last_peak_time = None
rr_list = []  # list of last RR intervals in seconds
latest_breath_strength = 0.0
def process_ir_peak(ir_value):
    global last_ir, last_peak_time, rr_list, stress_model, latest_breath_strength
    now = ticks_ms()
    hr = None
    hrv_ms = None
    if ir_value > FINGER_IR_THRESHOLD and last_ir <= FINGER_IR_THRESHOLD:
        if last_peak_time is not None:
            dt_ms = ticks_diff(now, last_peak_time)
            dt_s = dt_ms / 1000.0
            if 0.25 < dt_s < 2.0:
                rr_list.append(dt_s)
                if len(rr_list) > 16:
                    rr_list.pop(0)
                hr_float = 60.0 / dt_s
                hr = int(hr_float)
                if len(rr_list) >= 3:
                    mean = sum(rr_list) / len(rr_list)
                    var = sum((x - mean) * (x - mean) for x in rr_list) / len(rr_list)
                    hrv_ms = int((var ** 0.5) * 1000.0)
                if stress_model and stress_model.available:
                    stress_model.observe(now, dt_s, hr_float, latest_breath_strength)
        last_peak_time = now
    last_ir = ir_value
    return hr, hrv_ms

# Read latest red/ir and return safely
def get_latest_sensor():
    try:
        # prefer reading FIFO directly
        red, ir = max30102.read_fifo()
        return red, ir
    except Exception:
        return 0, 0

# Gyro/belly movement check: detect delta on chosen accel axis (use abs change)
prev_breath_axis = None
def breath_movement_detected():
    moved, _, _ = detect_breath_motion()
    return moved



class ShardedRandomForest:
    def __init__(self, header_path, tree_dir):
        self.available = False
        self.features = []
        self.threshold = 0.5
        self.trees = []
        self._error = None
        try:
            with open(header_path, 'r') as f:
                header = json.load(f)
            self.features = header.get('features', [])
            self.threshold = header.get('decision_threshold', 0.5)
            pattern = header.get('tree_pattern', 'tree_{:03d}.json')
            shard_dir = tree_dir or header.get('shard_dir', '')
            estimators = int(header.get('n_estimators', 0))
            if estimators <= 0:
                raise ValueError('no estimators in model header')
            for idx in range(estimators):
                name = pattern.format(idx)
                path_name = self._join(shard_dir, name)
                with open(path_name, 'r') as tree_file:
                    tree_data = json.load(tree_file)
                nodes = tree_data.get('nodes')
                if not nodes:
                    raise ValueError('empty tree at index %d' % idx)
                self.trees.append(nodes)
            if not self.trees:
                raise ValueError('no trees loaded')
            self.available = True
        except Exception as exc:
            self._error = exc
            self.available = False

    def _join(self, directory, name):
        if not directory:
            return name
        if directory.endswith('/'):
            return directory + name
        return directory + '/' + name

    def predict_proba(self, feature_map):
        if not self.available or not self.trees:
            return None
        total = 0.0
        for nodes in self.trees:
            total += self._eval_tree(nodes, feature_map)
        return total / len(self.trees)

    def _eval_tree(self, nodes, feature_map):
        idx = 0
        while True:
            node = nodes[idx]
            feature_idx = node[0]
            if feature_idx == -1:
                return float(node[1])
            threshold = node[1]
            left_idx = node[2]
            right_idx = node[3]
            feature_name = self.features[feature_idx] if feature_idx < len(self.features) else None
            value = feature_map.get(feature_name, 0.0) if feature_name else 0.0
            idx = left_idx if value <= threshold else right_idx



class PersonalizedStressModel:
    def __init__(self, header_path=MODEL_HEADER_PATH, tree_dir=MODEL_TREE_DIR):
        self.rf = ShardedRandomForest(header_path, tree_dir)
        self.available = self.rf.available
        self.history = []  # (time_ms, rr_s, hr_bpm)
        self.motion_history = []  # (time_ms, motion_level)
        self.last_probability = None
        self.last_timestamp_ms = 0
        self.threshold = self.rf.threshold if self.available else None
        self.last_features = None

    def observe(self, timestamp_ms, rr_interval_s, hr_bpm, motion_level=0.0):
        if not self.available:
            return
        self.history.append((timestamp_ms, rr_interval_s, hr_bpm))
        self.motion_history.append((timestamp_ms, motion_level))
        self._prune(timestamp_ms)
        if len(self.history) >= MODEL_MIN_BEATS:
            feature_map = self._build_features()
            probability = self.rf.predict_proba(feature_map)
            if probability is not None:
                self.last_probability = probability
                self.last_timestamp_ms = timestamp_ms
                self.last_features = feature_map

    def _prune(self, current_ms):
        window_start = current_ms - MODEL_WINDOW_MS
        while self.history and self.history[0][0] < window_start:
            self.history.pop(0)
        while self.motion_history and self.motion_history[0][0] < window_start:
            self.motion_history.pop(0)

    def _build_features(self):
        beats = self.history
        feature_names = self.rf.features
        data = {name: 0.0 for name in feature_names}
        if not beats:
            return data
        rr_values = [b[1] for b in beats]
        hr_values = [b[2] for b in beats]
        data['hr_mean_bpm'] = _mean(hr_values)
        data['sdnn_ms'] = _std(rr_values) * 1000.0
        data['rmssd_ms'] = _rmssd(rr_values) * 1000.0
        data['pnn20'] = _pnn(rr_values, 0.02)
        data['cv_ibi'] = _cv(rr_values)
        data['amp_cv'] = _amp_cv(hr_values)
        data['hr_var_60s'] = _variance(hr_values)
        data['hr_slope_60s'] = _hr_slope(beats)
        last_ms = beats[-1][0]
        recent_rr = [b[1] for b in beats if last_ms - b[0] <= 30000]
        recent_hr = [b[2] for b in beats if last_ms - b[0] <= 30000]
        if recent_rr:
            data['sdnn30_ms'] = _std(recent_rr) * 1000.0
            data['rmssd30_ms'] = _rmssd(recent_rr) * 1000.0
        motion_levels = [m[1] for m in self.motion_history]
        data['acc_vrms'] = _rms(motion_levels)
        data['amp_cv'] = _amp_cv(hr_values)
        data['step_hz'] = 0.0
        data['running'] = 1.0 if data['acc_vrms'] > 0.2 else 0.0
        return data

    def get_probability(self):
        if not self.available or self.last_probability is None:
            return None
        age = ticks_diff(ticks_ms(), self.last_timestamp_ms)
        if age > MODEL_MAX_PROB_AGE_MS:
            return None
        return self.last_probability

    def is_calm(self):
        prob = self.get_probability()
        if prob is None:
            return None
        return prob < self.threshold


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values):
    if not values:
        return 0.0
    mu = _mean(values)
    return _mean([(v - mu) * (v - mu) for v in values])


def _std(values):
    return math.sqrt(_variance(values)) if values else 0.0


def _rmssd(rr_values):
    if len(rr_values) < 2:
        return 0.0
    diffs = []
    for idx in range(1, len(rr_values)):
        diffs.append(rr_values[idx] - rr_values[idx - 1])
    if not diffs:
        return 0.0
    return math.sqrt(_mean([d * d for d in diffs]))


def _pnn(rr_values, threshold):
    if len(rr_values) < 2:
        return 0.0
    count = 0
    total = 0
    for idx in range(1, len(rr_values)):
        total += 1
        if abs(rr_values[idx] - rr_values[idx - 1]) > threshold:
            count += 1
    if total == 0:
        return 0.0
    return count / total


def _cv(values):
    if not values:
        return 0.0
    mu = _mean(values)
    if mu == 0.0:
        return 0.0
    return _std(values) / mu


def _amp_cv(hr_values):
    if not hr_values:
        return 0.0
    mu = _mean(hr_values)
    if mu == 0.0:
        return 0.0
    return (max(hr_values) - min(hr_values)) / mu


def _hr_slope(beats):
    if len(beats) < 2:
        return 0.0
    t0, _, hr0 = beats[0]
    tn, _, hrn = beats[-1]
    dt = (tn - t0) / 1000.0
    if dt <= 0:
        return 0.0
    return (hrn - hr0) / dt


def _rms(values):
    if not values:
        return 0.0
    return math.sqrt(_mean([v * v for v in values]))



stress_model = PersonalizedStressModel(MODEL_HEADER_PATH, MODEL_TREE_DIR)
if stress_model.available:
    print("Personalized stress model loaded.")
else:
    print("Stress model unavailable. Using heuristic thresholds.")

# DFPlayer helpers
def wait_for_track_or_skip():
    # Wait while busy; but listen for short button taps = skip
    while True:
        busy = player.queryBusy()
        if busy is None:
            # no busy pin, fallback to small wait
            sleep(0.2)
        else:
            if not busy:
                break
        # check for short button tap to skip:
        if not btn_pin.value():
            # button pressed: short click vs long press handled elsewhere
            # Debounce and consume short press: if released within 800ms -> skip
            t0 = ticks_ms()
            while not btn_pin.value():
                if ticks_diff(ticks_ms(), t0) > LONG_PRESS_MS:
                    # long press requested, handled at main loop
                    break
                sleep(0.05)
            # if released quickly, skip
            if ticks_diff(ticks_ms(), t0) < LONG_PRESS_MS:
                player.stop()
                show_centered("Skipped", "", "")
                sleep(0.5)
                break
        sleep(0.05)

# Button handling: returns (short, long) booleans
def read_button_event():
    # non-blocking check
    if not btn_pin.value():  # pressed (LOW)
        t0 = ticks_ms()
        # wait for release or long press
        while not btn_pin.value():
            if ticks_diff(ticks_ms(), t0) > LONG_PRESS_MS:
                # long press
                # wait release
                while not btn_pin.value():
                    sleep(0.05)
                return False, True
            sleep(0.02)
        # released within LONG_PRESS_MS: short press
        return True, False
    return False, False

# -----------------------------
# Greeting & prestart check
# -----------------------------
oled.fill(0)
oled.text("Hello — MediON", 0, 0)
oled.text("Place finger on", 0, 12)
oled.text("sensor and plug", 0, 24)
oled.text("headphones. Press", 0, 36)
oled.text("button to start.", 0, 48)
oled.show()

# Wait for finger + user confirmation (button)
start_confirmed = False
finger_ok = False
while True:
    finger_ok, r, ir = finger_present()
    # show finger + IR preview
    oled.fill_rect(0, 0, 128, 10, 0)
    oled.text("IR:%d" % ir, 0, 0)
    if not finger_ok:
        oled.fill_rect(0, 12, 128, 44, 0)
        oled.text("Put finger on", 0, 12)
        oled.text("sensor...", 0, 24)
        oled.text("Press button when", 0, 36)
        oled.text("ready", 0, 48)
        oled.show()
    else:
        oled.fill_rect(0, 12, 128, 44, 0)
        oled.text("Finger OK", 0, 12)
        oled.text("Press button to", 0, 24)
        oled.text("start", 0, 36)
        oled.show()

    short, longp = read_button_event()
    if longp:
        # user wants to stop before start: just show message and sleep
        oled_clear()
        oled.text("Startup aborted", 0, 0)
        oled.show()
        sleep(0.5)
        continue
    if short and finger_ok:
        start_confirmed = True
        break
    sleep(0.1)

# -----------------------------
# Session state variables
# -----------------------------
attempts = 0
session_active = True
breathing_retry_count = 0

# Always play 0001.mp3 at session start (breathing guided)
def play_and_monitor(track_num):
    """Play track and allow skip via short button, long press to abort session.
       Returns: 'skipped' if user skipped, 'stopped' if long press, 'done' otherwise."""
    player.playRoot(track_num)
    # Monitor busy + button events
    while True:
        busy = player.queryBusy()
        # check long press
        short, longp = read_button_event()
        if longp:
            player.stop()
            return 'stopped'
        if short:
            # skip
            player.stop()
            return 'skipped'
        if busy is None:
            # no busy pin: use a safe sleep and assume it will eventually finish
            sleep(0.1)
            # fallback: break when DFPlayer not responding? We'll rely on play durations
        else:
            if not busy:
                return 'done'
        sleep(0.05)

# Small helper to get current HR & HRV from rr_list
def get_current_hr_hrv():
    if len(rr_list) == 0:
        return None, None
    # latest HR from last RR
    last_rr = rr_list[-1]
    hr = int(60.0 / last_rr)
    hrv_ms = None
    if len(rr_list) >= 3:
        mean = sum(rr_list)/len(rr_list)
        var = sum((x - mean)**2 for x in rr_list)/len(rr_list)
        hrv_ms = int((var**0.5) * 1000.0)
    return hr, hrv_ms

# -----------------------------
# MAIN SESSION FLOW
# -----------------------------
show_centered("Starting session...", "", "")
sleep(0.5)

# always play breathing file 0001.mp3
res = play_and_monitor(1)
if res == 'stopped':
    show_centered("Stopped by user", "", "")
    session_active = False

# Continue if session is still active
while session_active:
    # sample a few IR values to update internal RR detection
    for _ in range(10):
        red, ir = get_latest_sensor()
        hr_peak, hrv_peak = process_ir_peak(ir)
        # small live update on OLED top
        curr_hr, curr_hrv = get_current_hr_hrv()
        oled.fill_rect(0,0,128,10,0)
        if curr_hr:
            oled.text("HR:{} HRV:{}ms".format(curr_hr, curr_hrv if curr_hrv else 0), 0, 0)
        else:
            oled.text("HR: -- HRV: --", 0, 0)
        oled.show()
        sleep(0.1)

    # Evaluate HR and HRV
    curr_hr, curr_hrv = get_current_hr_hrv()
    # handle None: not enough beats yet -> treat as not calm
    calm = False
    stress_prob = None
    if stress_model and stress_model.available:
        stress_prob = stress_model.get_probability()
    if stress_prob is not None:
        calm = stress_prob < stress_model.threshold
    elif curr_hr is not None and curr_hrv is not None:
        if curr_hr <= HR_CALM and curr_hrv >= HRV_CALM_MS:
            calm = True

    if calm:
        # play 0002.mp3 (sensation at feet)
        oled.text("Calm detected -> 0002", 0, 12)
        oled.show()
        res = play_and_monitor(2)
        if res == 'stopped':
            show_centered("Stopped by user", "", "")
            break
    else:
        # Not calm: try breathing retry up to MAX_RETRIES
        breathing_retry_count = 0
        improved = False
        while breathing_retry_count < MAX_RETRIES and not improved:
            breathing_retry_count += 1
            oled.text("Retry breathing: %d/%d" % (breathing_retry_count, MAX_RETRIES), 0, 12)
            oled.show()
            # Play 0003.mp3 (retry breathing)
            res = play_and_monitor(3)
            if res == 'stopped':
                show_centered("Stopped by user", "", "")
                session_active = False
                break
            # While 0003 played, we can check movement; if not enough, prompt with 0005
            # Quick check for movement now (a few samples)
            moved = False
            for _ in range(12):
                if breath_movement_detected():
                    moved = True
                    break
                # if user presses skip or stop during this time
                s, l = read_button_event()
                if l:
                    player.stop()
                    session_active = False
                    break
                if s:
                    player.stop()
                    break
                sleep(0.1)
            if not moved:
                # Play 0005 prompt about breathing movement
                oled.text("No belly movement -> 0005", 0, 24)
                oled.show()
                r = play_and_monitor(5)
                if r == 'stopped':
                    session_active = False
                    break
            # After retry, sample HR/HRV for improvement
            for _ in range(20):
                red, ir = get_latest_sensor()
                process_ir_peak(ir)
                sleep(0.1)
            curr_hr, curr_hrv = get_current_hr_hrv()
            stress_prob_retry = None
            if stress_model and stress_model.available:
                stress_prob_retry = stress_model.get_probability()
            if stress_prob_retry is not None:
                if stress_prob_retry < stress_model.threshold:
                    improved = True
                    break
            elif curr_hr is not None and curr_hrv is not None:
                if curr_hr <= HR_CALM and curr_hrv >= HRV_CALM_MS:
                    improved = True
                    break
        if not session_active:
            break
        if improved:
            # Play 0004 ("now continue by breathing normally")
            oled.text("Improved -> 0004", 0, 12)
            oled.show()
            r = play_and_monitor(4)
            if r == 'stopped':
                session_active = False
                break
        else:
            # After retries and still not improved -> proceed anyway to focus prompts (demo)
            oled.text("Still tense: proceed to focus", 0, 12)
            oled.show()
            sleep(0.5)

    # After breathing step and possible 0004 -> do muscle focus
    # For proof-of-concept we play 0006 (back muscles). Allow skip to 0007 for face.
    oled.text("Now focus on back. (skip->face)", 0, 24)
    oled.show()
    r = play_and_monitor(6)
    if r == 'stopped':
        session_active = False
        break
    # If user pressed skip while 0006, they asked to go to 0007 (face)
    # We interpret the 'skipped' return as user's desire to move on to 0007:
    if r == 'skipped':
        oled.text("Skipping to face focus -> 0007", 0, 24)
        oled.show()
        r2 = play_and_monitor(7)
        if r2 == 'stopped':
            session_active = False
            break

    # Check end condition: if HR and HRV are now calm enough end session with 0008
    for _ in range(20):
        red, ir = get_latest_sensor()
        process_ir_peak(ir)
        sleep(0.1)
    curr_hr, curr_hrv = get_current_hr_hrv()
    stress_prob_final = None
    if stress_model and stress_model.available:
        stress_prob_final = stress_model.get_probability()
    if stress_prob_final is not None:
        if stress_prob_final < stress_model.threshold:
            oled.text("Session complete -> 0008", 0, 12)
            oled.show()
            play_and_monitor(8)
            session_active = False
            break
    elif curr_hr is not None and curr_hrv is not None:
        if curr_hr <= HR_END and curr_hrv >= HRV_END_MS:
            oled.text("Session complete -> 0008", 0, 12)
            oled.show()
            play_and_monitor(8)
            session_active = False
            break

    # If not yet ending, allow loop to repeat (could go into another breathing round or end)
    # For demo: break to avoid infinite loop unless user wants extended session
    # We'll loop once more; otherwise it would keep repeating. To keep it simple, end here.
    oled.text("Session stage complete. End.", 0, 12)
    oled.show()
    session_active = False
    break

# FINALIZE
player.stop()
oled_clear()
oled.text("Session ended", 0, 0)
oled.show()
