# Using the display and basics of micropython on the Pico:
# https://dronebotworkshop.com/pi-pico/
# (check under "Adding a Display" to install a package for the ssd1306):
# - connect the Raspberry Pi Pico to your computer
# - choose Tools > Manage packages
# - search for "ssd1306" and install that

from machine import Pin, I2C
from time import sleep

# I2C pin setting: use GP2 and GP3 on I2C1:
sda=Pin(2)
scl=Pin(3)

# use I2C bus 1:
i2c=I2C(1, sda=sda, scl=scl, freq=400000)

from ssd1306 import SSD1306_I2C
oled = SSD1306_I2C(128, 64, i2c)

print(i2c.scan())

oled.text('Welcome to the', 0, 0)
oled.text('Pi Pico', 0, 10)
oled.text('Display Demo', 0, 20)
oled.show()
sleep(2)

oled.fill(1)
oled.show()
sleep(1)
oled.fill(0)
oled.show()

while True:
    oled.text("Hello World",0,0)
    for i in range (0, 164):
        oled.scroll(1,0)
        oled.show()
        sleep(0.01)