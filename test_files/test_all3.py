from machine import Pin, I2C, UART
from utime import sleep, ticks_ms

# === OLED DISPLAY SETUP ===
from ssd1306 import SSD1306_I2C

i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
oled = SSD1306_I2C(128, 64, i2c)
oled.fill(0)
oled.text("Initializing...", 0, 0)
oled.show()

# === DFPLAYER SETUP ===
from dfplayer import DFPlayer

player = DFPlayer(1, txPin=4, rxPin=5, busyPin=17)
player.setVolume(25)
oled.text("DFPlayer ready", 0, 10)
oled.show()
sleep(1)

print("Playing 0001.mp3")
player.playRoot(1)
for _ in range(5):
    print("BUSY =", player.queryBusy())
    sleep(1)
player.stop()

# === MAX30102 SETUP ===
from max30102 import MAX30102

max30102 = MAX30102(i2c_bus=0, sda=0, scl=1)
sleep(1)
oled.text("MAX30102 OK", 0, 20)
oled.show()
sleep(1)

# === MPU6050 SETUP ===
from mpu6050 import mpu6050

mpu = mpu6050(i2c)
print("MPU6050 readings:")
for _ in range(5):
    accel = mpu.get_accel_data()
    gyro = mpu.get_gyro_data()
    temp = mpu.get_temp()
    print("Accel:", accel, "| Gyro:", gyro, "| Temp:", temp)
    sleep(1)

# === HEART RATE & HRV GRAPH ===
# --- Graph Settings ---
OLED_WIDTH = 128
OLED_HEIGHT = 64
GRAPH_HEIGHT = 40          # reduced graph height
GRAPH_TOP = 20             # start graph below top text
GRAPH_BOTTOM = GRAPH_TOP + GRAPH_HEIGHT - 1
graph_data = [0] * OLED_WIDTH

def draw_graph(value):
    """Update scrolling graph without overwriting top text"""
    global graph_data
    val = min(max(int((value / 150000) * GRAPH_HEIGHT), 0), GRAPH_HEIGHT-1)
    graph_data = graph_data[1:] + [val]

    # Clear only graph area
    oled.fill_rect(0, GRAPH_TOP, OLED_WIDTH, GRAPH_HEIGHT, 0)

    # Draw graph line
    for x in range(OLED_WIDTH-1):
        y1 = GRAPH_BOTTOM - graph_data[x]
        y2 = GRAPH_BOTTOM - graph_data[x+1]
        oled.line(x, y1, x+1, y2, 1)

    # Optional: draw graph border
    oled.rect(0, GRAPH_TOP, OLED_WIDTH, GRAPH_HEIGHT, 1)
    oled.show()


oled.fill(0)
oled.text("Heart Rate Graph", 0, 0)
oled.show()

WIDTH = 128
HEIGHT = 64
graph_y = HEIGHT - 1  # bottom of graph
buffer_size = WIDTH
red_buf_graph = [0] * buffer_size

# Simple peak detection variables
last_ir = 0
last_peak_time = ticks_ms()
hr_list = []

def draw_graph(value):
    """Shift graph left and add new value"""
    global red_buf_graph
    # Scale value to fit OLED height (0-63)
    val = min(max(int((value / 100000) * HEIGHT), 0), HEIGHT-1)
    red_buf_graph = red_buf_graph[1:] + [val]

    oled.fill_rect(0, 10, WIDTH, HEIGHT-10, 0)  # clear graph area
    for x in range(WIDTH-1):
        oled.line(x, graph_y - red_buf_graph[x], x+1, graph_y - red_buf_graph[x+1], 1)
    oled.show()

def detect_hr(ir_value):
    """Detect heart beat peaks and compute HR & HRV"""
    global last_ir, last_peak_time, hr_list
    threshold = 50000  # adjust for your signal
    now = ticks_ms()
    hr = None
    hrv = None

    # Simple rising edge detection
    if ir_value > threshold and last_ir <= threshold:
        dt = (now - last_peak_time) / 1000  # seconds
        last_peak_time = now
        if dt > 0:
            hr = int(60 / dt)
            hr_list.append(dt)
            if len(hr_list) > 10:  # keep last 10 intervals
                hr_list.pop(0)
            # compute simple HRV = std deviation of last 10 RR intervals
            mean_rr = sum(hr_list)/len(hr_list)
            hrv = int((sum([(x - mean_rr)**2 for x in hr_list])/len(hr_list))**0.5 * 1000)  # ms
    last_ir = ir_value
    return hr, hrv

print("Starting live HR & HRV graph...")
while True:
    red, ir = max30102.read_fifo()
    draw_graph(ir)
    hr, hrv = detect_hr(ir)
    if hr is not None:
        oled.fill_rect(0, 0, 128, 10, 0)  # clear top text
        oled.text("HR:{}bpm HRV:{}ms".format(hr, hrv if hrv else 0), 0, 0)
        oled.show()