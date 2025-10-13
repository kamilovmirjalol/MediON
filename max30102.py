from machine import I2C, Pin
from utime import sleep

# I2C address
MAX30102_ADDR = 0x57

# Register addresses
REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03
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
    def __init__(self, i2c_bus=0, sda=0, scl=1):
        # Initialize I2C
        self.i2c = I2C(i2c_bus, sda=Pin(sda), scl=Pin(scl), freq=400000)
        self.address = MAX30102_ADDR

        self.reset()
        sleep(1)
        self.setup()

    # --- Low-level I2C helpers ---
    def write_reg(self, reg, val):
        self.i2c.writeto_mem(self.address, reg, bytes([val]))

    def read_reg(self, reg, nbytes=1):
        return self.i2c.readfrom_mem(self.address, reg, nbytes)

    # --- Sensor control ---
    def reset(self):
        self.write_reg(REG_MODE_CONFIG, 0x40)
        sleep(1)

    def setup(self):
        self.write_reg(REG_INTR_ENABLE_1, 0xc0)
        self.write_reg(REG_INTR_ENABLE_2, 0x00)
        self.write_reg(REG_FIFO_WR_PTR, 0x00)
        self.write_reg(REG_OVF_COUNTER, 0x00)
        self.write_reg(REG_FIFO_RD_PTR, 0x00)
        self.write_reg(REG_FIFO_CONFIG, 0x4f)
        self.write_reg(REG_MODE_CONFIG, 0x03)    # SpO2 mode
        self.write_reg(REG_SPO2_CONFIG, 0x27)
        self.write_reg(REG_LED1_PA, 0x24)
        self.write_reg(REG_LED2_PA, 0x24)
        self.write_reg(REG_PILOT_PA, 0x7f)

    def get_data_present(self):
        read_ptr = self.read_reg(REG_FIFO_RD_PTR)[0]
        write_ptr = self.read_reg(REG_FIFO_WR_PTR)[0]
        if read_ptr == write_ptr:
            return 0
        else:
            num_samples = write_ptr - read_ptr
            if num_samples < 0:
                num_samples += 32
            return num_samples

    def read_fifo(self):
        # Clear interrupts
        _ = self.read_reg(REG_INTR_STATUS_1)
        _ = self.read_reg(REG_INTR_STATUS_2)

        d = self.read_reg(REG_FIFO_DATA, 6)
        red = ((d[0] << 16) | (d[1] << 8) | d[2]) & 0x03FFFF
        ir = ((d[3] << 16) | (d[4] << 8) | d[5]) & 0x03FFFF
        return red, ir

    def read_sequential(self, amount=100):
        red_buf = []
        ir_buf = []
        count = amount

        while count > 0:
            num_bytes = self.get_data_present()
            while num_bytes > 0 and count > 0:
                red, ir = self.read_fifo()
                red_buf.append(red)
                ir_buf.append(ir)
                num_bytes -= 1
                count -= 1

        return red_buf, ir_buf
