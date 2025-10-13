from machine import Pin, I2C, UART
from utime import sleep, sleep_ms

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

# Test DFPlayer root playback
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
red_buf, ir_buf = max30102.read_sequential(5)
print("MAX30102 sample readings:")
for r, ir in zip(red_buf, ir_buf):
    print("Red:", r, "IR:", ir)
oled.text("MAX30102 OK", 0, 20)
oled.show()
sleep(1)

# === MPU6050 SETUP ===
from mpu6050 import mpu6050  # MicroPython version only, no smbus

mpu = mpu6050(i2c)
print("MPU6050 readings:")
for _ in range(5):
    accel = mpu.get_accel_data()
    gyro = mpu.get_gyro_data()
    temp = mpu.get_temp()
    print("Accel:", accel, "| Gyro:", gyro, "| Temp:", temp)
    sleep(1)

oled.fill(0)
oled.text("All tests done!", 0, 0)
oled.show()
