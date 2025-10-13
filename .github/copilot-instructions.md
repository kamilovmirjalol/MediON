# Copilot Instructions for MediON

## Project Overview
This project is a minimal MicroPython/Pico project for controlling an onboard LED. The main script is `blink.py`, which toggles the LED on and off every second using the `machine` and `utime` modules.

## Key Files
- `blink.py`: Main entry point. Contains a simple infinite loop to blink the onboard LED. Uses `Pin("LED", Pin.OUT)` for hardware abstraction.
- `.micropico/`: (If present) May contain MicroPython or Pico-specific configuration or deployment scripts.
- `.vscode/`: (If present) May contain VS Code settings or launch configurations.

## Development Workflow
- **Run on device:** Deploy `blink.py` to a MicroPython-compatible device (e.g., Raspberry Pi Pico) using your preferred tool (e.g., Thonny, ampy, rshell, or VS Code extensions).
- **Interrupting:** The script handles `KeyboardInterrupt` to allow safe stopping during development.
- **No build system:** There is no build or test automation. All logic is in `blink.py`.

## Coding Conventions
- Use MicroPython APIs (`machine.Pin`, `utime.sleep`).
- Use `Pin("LED", Pin.OUT)` for the onboard LED (works on Pico and similar boards).
- Keep scripts minimal and hardware-focused.
- Use try/except for graceful interruption.

## Example Pattern
```python
from machine import Pin
from utime import sleep

pin = Pin("LED", Pin.OUT)
while True:
    try:
        pin.toggle()
        sleep(1)
    except KeyboardInterrupt:
        break
pin.off()
```

## AI Agent Guidance
- Focus on hardware control and MicroPython idioms.
- Avoid adding unnecessary abstractions or dependencies.
- If adding new scripts, follow the minimal, direct style of `blink.py`.
- Document any new hardware pin usage clearly in code comments.

---
If you add new workflows, scripts, or conventions, update this file to keep AI agents productive.