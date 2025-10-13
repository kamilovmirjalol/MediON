from MAX30102 import MAX30102
from utime import sleep

sensor = MAX30102(i2c_bus=0, sda=0, scl=1)

# Read 10 samples
red_vals, ir_vals = sensor.read_sequential(40)
for i in range(40):
    print("Red:", red_vals[i], "IR:", ir_vals[i])
    sleep(0.5)
