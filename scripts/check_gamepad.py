"""Interactive gamepad check for the LeRobot gym_hil recording setup.

Run this before a recording session to confirm that the controller is visible
to pygame and that button/axis numbers match what the env expects.

    .venv/bin/python scripts/check_gamepad.py
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import pygame
    except ImportError:
        print(
            "pygame is not installed. Install the collection extras: uv pip install -e '.[collect]'"
        )
        return 2

    pygame.init()
    pygame.joystick.init()
    try:
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Check the USB connection and udev permissions.")
            return 1

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Connected gamepad: {joystick.get_name()}")
        print("Press buttons or move sticks to inspect mappings (Ctrl+C to exit).\n")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"button down -> id={event.button}")
                elif event.type == pygame.JOYAXISMOTION and abs(event.value) > 0.2:
                    print(f"axis motion  -> axis={event.axis} value={event.value:.2f}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nGamepad check finished.")
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
