from machine import Pin
import time

# GP15 with internal pull-up
button = Pin(15, Pin.IN, Pin.PULL_UP)

while True:
    if not button.value():  # pressed = 0
        print("Button pressed!")
    time.sleep(0.3)
