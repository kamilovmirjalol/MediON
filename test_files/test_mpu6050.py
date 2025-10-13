from machine import I2C, Pin
from time import sleep

i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)
MPU_ADDR = 0x68

# Registers
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B

# Wake up MPU6050
i2c.writeto_mem(MPU_ADDR, PWR_MGMT_1, bytes([0]))

def bytes_to_int(high, low):
    val = (high << 8) | low
    if val >= 0x8000:
        val -= 0x10000
    return val

def read_raw():
    data = i2c.readfrom_mem(MPU_ADDR, ACCEL_XOUT_H, 14)
    ax = bytes_to_int(data[0], data[1])
    ay = bytes_to_int(data[2], data[3])
    az = bytes_to_int(data[4], data[5])
    t_raw = bytes_to_int(data[6], data[7])
    gx = bytes_to_int(data[8], data[9])
    gy = bytes_to_int(data[10], data[11])
    gz = bytes_to_int(data[12], data[13])
    return ax, ay, az, t_raw, gx, gy, gz

def convert(ax, ay, az, t_raw, gx, gy, gz):
    ax /= 16384.0
    ay /= 16384.0
    az /= 16384.0
    temp = (t_raw / 340.0) + 36.53
    gx /= 131.0
    gy /= 131.0
    gz /= 131.0
    return ax, ay, az, temp, gx, gy, gz

print("Reading MPU6050 (Ctrl+C to stop)")

try:
    while True:
        ax, ay, az, t_raw, gx, gy, gz = read_raw()
        ax, ay, az, temp, gx, gy, gz = convert(ax, ay, az, t_raw, gx, gy, gz)
        print("Accel: x={:.2f}g y={:.2f}g z={:.2f}g | Temp: {:.2f}°C | Gyro: x={:.2f}°/s y={:.2f}°/s z={:.2f}°/s"
              .format(ax, ay, az, temp, gx, gy, gz))
        sleep(0.5)
except KeyboardInterrupt:
    print("Stopped by user")
