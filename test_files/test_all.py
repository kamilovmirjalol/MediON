from machine import Pin, UART, I2C
from utime import sleep, sleep_ms
from ssd1306 import SSD1306_I2C

# -------------------------------
# Initialize I2C and OLED
# -------------------------------
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
oled = SSD1306_I2C(128, 64, i2c)

# -------------------------------
# DFPlayer driver
# -------------------------------
class DFPlayer:
    UART_BAUD_RATE = 9600
    UART_BITS = 8
    UART_PARITY = None
    UART_STOP = 1
    START_BYTE = 0x7E
    VERSION_BYTE = 0xFF
    COMMAND_LENGTH = 0x06
    ACKNOWLEDGE = 0x01
    END_BYTE = 0xEF
    COMMAND_LATENCY = 200

    def __init__(self, uartInstance, txPin, rxPin, busyPin=None):
        self.uart = UART(uartInstance, baudrate=self.UART_BAUD_RATE,
                         tx=Pin(txPin), rx=Pin(rxPin),
                         bits=self.UART_BITS, parity=self.UART_PARITY, stop=self.UART_STOP)
        if busyPin is not None:
            self.playerBusy = Pin(busyPin, Pin.IN, Pin.PULL_UP)
        else:
            self.playerBusy = None

    def split(self, num):
        return num >> 8, num & 0xFF

    def sendcmd(self, command, parameter1, parameter2):
        checksum = -(self.VERSION_BYTE + self.COMMAND_LENGTH + command + self.ACKNOWLEDGE + parameter1 + parameter2)
        highByte, lowByte = self.split(checksum)
        toSend = bytes([
            self.START_BYTE, self.VERSION_BYTE, self.COMMAND_LENGTH,
            command, self.ACKNOWLEDGE, parameter1, parameter2,
            highByte & 0xFF, lowByte & 0xFF, self.END_BYTE
        ])
        self.uart.write(toSend)
        sleep_ms(self.COMMAND_LATENCY)

    def setVolume(self, volume):
        self.sendcmd(0x06, 0x00, volume)

    def playRoot(self, index):
        self.sendcmd(0x03, 0x00, index)

# -------------------------------
# MAX30102 driver
# -------------------------------
MAX30102_ADDR = 0x57
REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_PILOT_PA = 0x10

class MAX30102:
    def __init__(self, i2c_bus):
        self.i2c = i2c_bus
        self.address = MAX30102_ADDR
        self.reset()
        sleep(1)
        self.setup()

    def write_reg(self, reg, val):
        self.i2c.writeto_mem(self.address, reg, bytes([val]))

    def read_reg(self, reg, nbytes=1):
        return self.i2c.readfrom_mem(self.address, reg, nbytes)

    def reset(self):
        self.write_reg(REG_MODE_CONFIG, 0x40)
        sleep(1)

    def setup(self):
        self.write_reg(REG_INTR_STATUS_1, 0xC0)
        self.write_reg(REG_INTR_STATUS_2, 0x00)
        self.write_reg(REG_FIFO_WR_PTR, 0x00)
        self.write_reg(REG_OVF_COUNTER, 0x00)
        self.write_reg(REG_FIFO_RD_PTR, 0x00)
        self.write_reg(REG_FIFO_CONFIG, 0x4F)
        self.write_reg(REG_MODE_CONFIG, 0x03)
        self.write_reg(REG_SPO2_CONFIG, 0x27)
        self.write_reg(REG_LED1_PA, 0x24)
        self.write_reg(REG_LED2_PA, 0x24)
        self.write_reg(REG_PILOT_PA, 0x7F)

    def get_data_present(self):
        read_ptr = self.read_reg(REG_FIFO_RD_PTR)[0]
        write_ptr = self.read_reg(REG_FIFO_WR_PTR)[0]
        num_samples = (write_ptr - read_ptr) % 32
        return num_samples

    def read_fifo(self):
        _ = self.read_reg(REG_INTR_STATUS_1)
        _ = self.read_reg(REG_INTR_STATUS_2)
        d = self.read_reg(REG_FIFO_DATA, 6)
        red = ((d[0] << 16) | (d[1] << 8) | d[2]) & 0x03FFFF
        ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
        return red, ir

# -------------------------------
# MPU6050 driver (MicroPython version)
# -------------------------------
class MPU6050:
    GRAVITY_MS2 = 9.80665
    ACCEL_XOUT0 = 0x3B
    ACCEL_YOUT0 = 0x3D
    ACCEL_ZOUT0 = 0x3F
    TEMP_OUT0   = 0x41
    GYRO_XOUT0  = 0x43
    GYRO_YOUT0  = 0x45
    GYRO_ZOUT0  = 0x47
    PWR_MGMT_1  = 0x6B

    def __init__(self, i2c_bus, addr=0x68):
        self.addr = addr
        self.i2c = i2c_bus
        self.i2c.writeto_mem(self.addr, self.PWR_MGMT_1, bytes([0]))

    def read_i2c_word(self, reg):
        high = self.i2c.readfrom_mem(self.addr, reg, 1)[0]
        low = self.i2c.readfrom_mem(self.addr, reg+1, 1)[0]
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        return val

    def get_accel_data(self):
        x = self.read_i2c_word(self.ACCEL_XOUT0) / 16384
        y = self.read_i2c_word(self.ACCEL_YOUT0) / 16384
        z = self.read_i2c_word(self.ACCEL_ZOUT0) / 16384
        return {'x': x, 'y': y, 'z': z}

    def get_gyro_data(self):
        x = self.read_i2c_word(self.GYRO_XOUT0) / 131
        y = self.read_i2c_word(self.GYRO_YOUT0) / 131
        z = self.read_i2c_word(self.GYRO_ZOUT0) / 131
        return {'x': x, 'y': y, 'z': z}

    def get_temp(self):
        raw_temp = self.read_i2c_word(self.TEMP_OUT0)
        return raw_temp / 340 + 36.53

# -------------------------------
# Initialize all devices
# -------------------------------
dfplayer = DFPlayer(uartInstance=1, txPin=4, rxPin=5)
dfplayer.setVolume(25)

max30102 = MAX30102(i2c)
mpu = MPU6050(i2c)

# -------------------------------
# Test loop
# -------------------------------
print("Playing 0001.mp3 ...")
dfplayer.playRoot(1)

for _ in range(20):
    # MAX30102
    if max30102.get_data_present() > 0:
        red, ir = max30102.read_fifo()
    else:
        red, ir = 0, 0

    # MPU6050
    accel = mpu.get_accel_data()
    gyro = mpu.get_gyro_data()
    temp = mpu.get_temp()

    # OLED display
    oled.fill(0)
    oled.text("Red:{} IR:{}".format(red, ir), 0, 0)
    oled.text("Acc:{:.2f},{:.2f},{:.2f}".format(accel['x'], accel['y'], accel['z']), 0, 10)
    oled.text("Gyro:{:.1f},{:.1f},{:.1f}".format(gyro['x'], gyro['y'], gyro['z']), 0, 20)
    oled.text("Tmp:{:.1f}C".format(temp), 0, 30)
    oled.show()
    sleep(1)

print("Test complete.")
