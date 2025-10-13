# mpu6050.py - MicroPython driver for MPU6050 on Raspberry Pi Pico
# Works with Raspberry Pi Pico / Pico W / Pico 2W
# Uses machine.I2C only, no smbus dependency

from machine import I2C, Pin
from utime import sleep_ms

class mpu6050:
    # MPU-6050 Registers
    PWR_MGMT_1   = 0x6B
    ACCEL_XOUT0  = 0x3B
    ACCEL_YOUT0  = 0x3D
    ACCEL_ZOUT0  = 0x3F
    TEMP_OUT0    = 0x41
    GYRO_XOUT0   = 0x43
    GYRO_YOUT0   = 0x45
    GYRO_ZOUT0   = 0x47
    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG  = 0x1B
    MPU_CONFIG   = 0x1A

    GRAVITY_MS2 = 9.80665

    # Scale modifiers
    ACCEL_SCALE_MODIFIER_2G = 16384.0
    ACCEL_SCALE_MODIFIER_4G = 8192.0
    ACCEL_SCALE_MODIFIER_8G = 4096.0
    ACCEL_SCALE_MODIFIER_16G = 2048.0

    GYRO_SCALE_MODIFIER_250DEG  = 131.0
    GYRO_SCALE_MODIFIER_500DEG  = 65.5
    GYRO_SCALE_MODIFIER_1000DEG = 32.8
    GYRO_SCALE_MODIFIER_2000DEG = 16.4

    # Pre-defined ranges
    ACCEL_RANGE_2G  = 0x00
    ACCEL_RANGE_4G  = 0x08
    ACCEL_RANGE_8G  = 0x10
    ACCEL_RANGE_16G = 0x18

    GYRO_RANGE_250DEG  = 0x00
    GYRO_RANGE_500DEG  = 0x08
    GYRO_RANGE_1000DEG = 0x10
    GYRO_RANGE_2000DEG = 0x18

    def __init__(self, i2c, address=0x68):
        self.i2c = i2c
        self.address = address
        # Wake up MPU6050 (clear sleep bit)
        self.i2c.writeto_mem(self.address, self.PWR_MGMT_1, bytes([0x00]))
        sleep_ms(100)

    # --- Low-level helpers ---
    def read_word(self, reg):
        data = self.i2c.readfrom_mem(self.address, reg, 2)
        val = (data[0] << 8) | data[1]
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    # --- Temperature ---
    def get_temp(self):
        raw = self.read_word(self.TEMP_OUT0)
        return (raw / 340.0) + 36.53

    # --- Accelerometer ---
    def get_accel_data(self, g=False):
        x = self.read_word(self.ACCEL_XOUT0)
        y = self.read_word(self.ACCEL_YOUT0)
        z = self.read_word(self.ACCEL_ZOUT0)

        # Assuming default 2G range
        x /= self.ACCEL_SCALE_MODIFIER_2G
        y /= self.ACCEL_SCALE_MODIFIER_2G
        z /= self.ACCEL_SCALE_MODIFIER_2G

        if not g:
            x *= self.GRAVITY_MS2
            y *= self.GRAVITY_MS2
            z *= self.GRAVITY_MS2

        return {'x': x, 'y': y, 'z': z}

    # --- Gyroscope ---
    def get_gyro_data(self):
        x = self.read_word(self.GYRO_XOUT0)
        y = self.read_word(self.GYRO_YOUT0)
        z = self.read_word(self.GYRO_ZOUT0)

        # Assuming default 250 deg/s range
        x /= self.GYRO_SCALE_MODIFIER_250DEG
        y /= self.GYRO_SCALE_MODIFIER_250DEG
        z /= self.GYRO_SCALE_MODIFIER_250DEG

        return {'x': x, 'y': y, 'z': z}

    # --- Combined ---
    def get_all_data(self, g=False):
        return {
            'accel': self.get_accel_data(g),
            'gyro': self.get_gyro_data(),
            'temp': self.get_temp()
        }

# --- Test block ---
if __name__ == "__main__":
    # Example I2C setup for Pico 2W
    from machine import Pin, I2C
    i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
    mpu = mpu6050(i2c)

    while True:
        data = mpu.get_all_data()
        print("Accel:", data['accel'], "| Gyro:", data['gyro'], "| Temp:", data['temp'])
