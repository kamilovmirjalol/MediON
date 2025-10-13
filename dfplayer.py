# DFPlayer MP3 player Driver for Raspberry Pi Pico
# Root-folder play version

from machine import UART, Pin
from utime import sleep_ms, sleep

class DFPlayer():
    UART_BAUD_RATE=9600
    UART_BITS=8
    UART_PARITY=None
    UART_STOP=1
    
    START_BYTE = 0x7E
    VERSION_BYTE = 0xFF
    COMMAND_LENGTH = 0x06
    ACKNOWLEDGE = 0x01
    END_BYTE = 0xEF
    COMMAND_LATENCY = 200  # shorter delay works fine

    def __init__(self, uartInstance, txPin, rxPin, busyPin=None):
        if busyPin is not None:
            self.playerBusy = Pin(busyPin, Pin.IN, Pin.PULL_UP)
        else:
            self.playerBusy = None
        self.uart = UART(
            uartInstance, 
            baudrate=self.UART_BAUD_RATE, 
            tx=Pin(txPin), 
            rx=Pin(rxPin), 
            bits=self.UART_BITS, 
            parity=self.UART_PARITY, 
            stop=self.UART_STOP
        )

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
        return self.uart.read()

    def queryBusy(self):
        if self.playerBusy:
            return not self.playerBusy.value()
        return None

    # Volume 0–30
    def setVolume(self, volume):
        self.sendcmd(0x06, 0x00, volume)

    def playRoot(self, index):
        """Play file in SD root folder by index.
        0001.mp3 = 1, 0002.mp3 = 2, etc."""
        self.sendcmd(0x03, 0x00, index)

    def stop(self):
        self.sendcmd(0x16, 0x00, 0x00)

    def pause(self):
        self.sendcmd(0x0E, 0x00, 0x00)

    def resume(self):
        self.sendcmd(0x0D, 0x00, 0x00)
