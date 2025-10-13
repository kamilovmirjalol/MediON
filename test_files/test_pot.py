from machine import ADC
import time

pot = ADC(26)  # GP26 = ADC0

while True:
    value = pot.read_u16()  # 0 to 65535
    print("Pot value:", value)
    time.sleep(0.2)

    