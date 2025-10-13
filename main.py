# main.py - MediON guided NSDR prototype

from machine import I2C, Pin
from utime import sleep, ticks_ms, ticks_diff

from ssd1306 import SSD1306_I2C
from dfplayer import DFPlayer
from max30102 import MAX30102
from mpu6050 import mpu6050

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
LONG_PRESS_MS = 2000
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
IR_BASELINE_ALPHA = 0.97
PEAK_RISE_TRIGGER = 120
PEAK_FALL_RESET = 40
MIN_BEAT_INTERVAL_MS = 320
MAX_BEAT_INTERVAL_MS = 2000
RISE_HYSTERESIS = 40
PEAK_DYNAMIC_FACTOR = 0.45
FALL_RATIO = 0.35

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
last_peak_ms = None
rr_samples = []
rr_window = []
current_hr = None
current_hr_smooth = None
current_hrv = None
last_rr_update_ms = 0

breath_direction = "Stable"
last_gyro_value = 0.0

skip_notice = ""
skip_notice_expires = 0
last_debug_print_ms = 0
last_signal_value = 0.0
prev_signal_value = 0.0
signal_envelope = 0.0
last_dynamic_trigger = 0
last_dynamic_fall = 0

breath_baseline = None
breath_signal = 0.0
breath_strength = 0.0
last_display_hr = None
last_display_hrv = None
last_display_update_ms = 0

ir_baseline = 0.0
peak_active = False


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
    global current_hr, current_hrv, current_hr_smooth, last_display_hr, last_display_hrv, last_display_update_ms
    if last_rr_update_ms == 0:
        return
    now = ticks_ms()
    age = ticks_diff(now, last_rr_update_ms)
    if age > STALE_HR_TIMEOUT_MS:
        current_hr = None
        if age > STALE_HR_TIMEOUT_MS + 2000:
            current_hr_smooth = None
        if age > DISPLAY_STALE_TIMEOUT_MS:
            last_display_hr = None
            last_display_hrv = None
    if last_display_update_ms:
        display_age = ticks_diff(now, last_display_update_ms)
        if display_age > DISPLAY_STALE_TIMEOUT_MS:
            last_display_hr = None
            last_display_hrv = None


def reset_hr_processing():
    global rr_samples, rr_window, current_hr, current_hrv, current_hr_smooth, last_rr_update_ms, last_peak_ms, ir_baseline, peak_active, signal_envelope, last_dynamic_trigger, last_dynamic_fall, breath_baseline, breath_signal, breath_strength, breath_direction, last_gyro_value
    rr_samples = []
    rr_window = []
    current_hr = None
    current_hr_smooth = None
    current_hrv = None
    last_rr_update_ms = 0
    last_peak_ms = None
    ir_baseline = 0.0
    peak_active = False
    signal_envelope = 0.0
    last_dynamic_trigger = 0
    last_dynamic_fall = 0
    breath_baseline = None
    breath_signal = 0.0
    breath_strength = 0.0
    breath_direction = "Stable"
    last_gyro_value = 0.0


def update_hr_state_from_ir(ir_value):
    global last_ir_value, last_peak_ms, rr_samples, rr_window, current_hr, current_hrv, current_hr_smooth, last_rr_update_ms, ir_baseline, peak_active, last_signal_value, prev_signal_value, signal_envelope, last_dynamic_trigger, last_dynamic_fall, last_display_hr, last_display_hrv, last_display_update_ms
    if ir_value is None:
        return
    now = ticks_ms()
    finger_present = ir_value >= FINGER_IR_THRESHOLD
    if not finger_present:
        reset_hr_processing()
        last_ir_value = ir_value
        return
    if ir_baseline == 0.0:
        ir_baseline = float(ir_value)
    else:
        ir_baseline = (IR_BASELINE_ALPHA * ir_baseline) + ((1.0 - IR_BASELINE_ALPHA) * ir_value)
    prev_signal_value = last_signal_value
    signal = ir_value - ir_baseline
    last_signal_value = signal
    signal_envelope = (signal_envelope * 0.9) + (abs(signal) * 0.1)
    dynamic_trigger = max(PEAK_RISE_TRIGGER, int(signal_envelope * PEAK_DYNAMIC_FACTOR))
    dynamic_fall = max(PEAK_FALL_RESET, int(dynamic_trigger * FALL_RATIO))
    last_dynamic_trigger = dynamic_trigger
    last_dynamic_fall = dynamic_fall
    rising = signal >= (prev_signal_value - RISE_HYSTERESIS)
    if not peak_active and signal >= dynamic_trigger and rising:
        if last_peak_ms is not None:
            interval = ticks_diff(now, last_peak_ms)
            if MIN_BEAT_INTERVAL_MS <= interval <= MAX_BEAT_INTERVAL_MS:
                if DEBUG:
                    print("[BEAT_CANDIDATE] signal:{} interval:{} ms".format(int(signal), interval))
                rr = interval / 1000.0
                rr_samples.append(rr)
                rr_window.append(rr)
                if len(rr_samples) > 16:
                    rr_samples.pop(0)
                if len(rr_window) > 12:
                    rr_window.pop(0)
                recent = rr_window[-3:] if len(rr_window) >= 3 else rr_window
                avg_rr = sum(recent) / len(recent)
                current_hr = int(60.0 / avg_rr)
                if current_hr_smooth is None:
                    current_hr_smooth = current_hr
                else:
                    current_hr_smooth = int((current_hr_smooth * 0.7) + (current_hr * 0.3))
                if DEBUG:
                    print("[BEAT] interval_ms:{} HR:{}".format(int(interval), current_hr))
                if len(rr_window) >= 3:
                    mean = sum(rr_window) / len(rr_window)
                    var = sum((x - mean) * (x - mean) for x in rr_window) / len(rr_window)
                    current_hrv = int((var ** 0.5) * 1000.0)
                    last_display_hrv = current_hrv
                last_display_hr = current_hr
                last_display_update_ms = now
                last_rr_update_ms = now
        last_peak_ms = now
        peak_active = True
    elif peak_active and signal <= dynamic_fall:
        peak_active = False
    last_ir_value = ir_value
def pull_sensor_sample(max_reads=PPG_BATCH_READS):
    global last_debug_print_ms
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
        update_hr_state_from_ir(ir)
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
            print("[DBG] IR:{} base:{} signal:{} dyn:{} fall:{} finger:{} HR:{} HRV:{} HP_level:{} HP_connected:{}".format(
                  ir_val if ir_val is not None else "None",
                  int(ir_baseline) if ir_baseline else 0,
                  int(last_signal_value),
                  last_dynamic_trigger,
                  last_dynamic_fall,
                  finger_state,
                  hr_display,
                  hrv_display,
                  hp_level,
                  hp_connected))
            last_debug_print_ms = now
    return red_val, ir_val


def finger_on_sensor(ir_value):
    if ir_value is None:
        return False
    return ir_value >= FINGER_IR_THRESHOLD



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

    abd = ("Breath: " + breath_direction)[:16]
    oled.text(abd, 0, 12)

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
    if skip_notice and ticks_diff(skip_notice_expires, now) > 0:
        oled.text(skip_notice[:16], 0, 56)
    elif warning:
        oled.text(warning[:16], 0, 56)
    oled.show()


def read_button_event():
    if button.value() == 0:
        start = ticks_ms()
        while button.value() == 0:
            if ticks_diff(ticks_ms(), start) >= LONG_PRESS_MS:
                while button.value() == 0:
                    sleep(0.02)
                return False, True
            sleep(0.02)
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
        detect_breath_motion()
        finger_ok = finger_on_sensor(ir)
        headphone_ok = headphones_connected()
        warning = compose_warning_line(finger_ok, headphone_ok)
        render_status(header, detail, warning)
        short, long_press = read_button_event()
        if long_press:
            set_skip_notice("Session stopped")
            return 'stop'
        if short:
            set_skip_notice("Button noted")
        sleep(IDLE_SAMPLE_DELAY)
    return 'ok'


def play_and_monitor(track, header="", detail="", monitor_breath=False, allow_skip=True):
    movement_samples = 0
    movement_hits = 0
    player.playRoot(track)
    while True:
        _, ir = pull_sensor_sample()
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
        if short:
            if allow_skip:
                player.stop()
                set_skip_notice("Skipped %04d" % track)
                return 'skipped', movement_ratio(movement_hits, movement_samples)
            else:
                set_skip_notice("Track %04d locked" % track)
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
    hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
    hrv_value = current_hrv if current_hrv is not None else last_display_hrv
    if hr_value is None or hrv_value is None:
        return False
    return hr_value <= HR_CALM and hrv_value >= HRV_CALM_MS


def end_condition_met():
    hr_value = current_hr_smooth if current_hr_smooth is not None else (current_hr if current_hr is not None else last_display_hr)
    hrv_value = current_hrv if current_hrv is not None else last_display_hrv
    if hr_value is None or hrv_value is None:
        return False
    return hr_value <= HR_END and hrv_value >= HRV_END_MS


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
    run_session()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        end_session("Session stopped")
