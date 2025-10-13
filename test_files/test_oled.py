from machine import Pin, I2C
import ssd1306
import time

# Initialize I2C on Pico
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)  
#i2c=I2C(1, sda=Pin(2), scl=Pin(3), freq=400000)
# Initialize OLED (128x64), address 0x3C
oled = ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C)

# Small delay to ensure display is ready
time.sleep(0.1)

# Clear display
oled.fill(0)
oled.show()

# Display welcome text
oled.text("Hello MediON!", 0, 0)
oled.text("OLED is working", 0, 16)
oled.show()

# Keep text visible for 3 seconds
time.sleep(3)

# Simple animation: moving a rectangle down the screen
for y in range(0, 56, 4):  # step every 4 pixels
    oled.fill(0)
    oled.text("Hello MediON!", 0, 0)
    oled.text("OLED is working", 0, 16)
    oled.rect(50, y, 20, 8, 1)  # x, y, width, height, color
    oled.show()
    time.sleep(0.2)

# Clear screen at the end
oled.fill(0)
oled.show()