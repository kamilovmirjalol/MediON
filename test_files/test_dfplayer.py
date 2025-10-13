from dfplayer import DFPlayer
from machine import Pin
import time

# Wiring: UART1 GP4=TX -> DFPlayer RX, GP5=RX -> DFPlayer TX
# BUSY pin wired to GP17
player = DFPlayer(1, txPin=4, rxPin=5, busyPin=17)

print("Initializing DFPlayer...")

# 1. Explicitly select SD card as source
player.sendcmd(0x09, 0x00, 0x01)  # setPlaybackSource(1) = TF/SD

# 2. Set volume (try mid-high)
player.setVolume(7)

player.sendcmd(0x0C, 0x00, 0x00)  # reset
time.sleep(2)
player.sendcmd(0x09, 0x00, 0x01)  # set source to TF
# 3. Play first file (0001.mp3 in root)
print("Playing 0001.mp3 ...")
player.playRoot(2)

# 4. Monitor BUSY pin while playing
for i in range(20):  # ~20 seconds
    if player.queryBusy() is not None:
        print("BUSY =", player.queryBusy())
    time.sleep(1)

print("Test done.")
