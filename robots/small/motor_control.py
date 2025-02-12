#!/usr/bin/env python3
"""
Standalone motor control script for the turbopi robot.
This script writes a two-byte I2C message to control a motor.
It uses the same basic protocol as the ROS-based C++ code:
  • The command byte is computed as the motor base address plus (motor_id - 1).
  • The speed value is computed from an "effort" value.
    - If effort is "zero" (i.e. floor(effort) == 0), speed remains 0.
    - Otherwise, speed is computed as: 
          speed = (ceil(effort) / 31) * 100,
      then clamped to ±100 and boosted to a minimum magnitude of 50,
      and finally inverted for motors whose id is odd.
Usage:
  python motor_control.py --motor 1 --effort 15
You can override the I²C bus number, device (i.e. slave) address, and motor base address.
  
Note: This script requires the "smbus" (or smbus2) Python module.  
For example, install via: pip install smbus2
"""

import math
import argparse

try:
    # Depending on your system, you might need to install and import smbus2 instead.
    import smbus
except ImportError:
    raise ImportError("smbus module not found. You might need to install smbus2.")

def compute_speed(effort, motor_id):
    """
    Mimics the C++ motor effort-to-speed calculation.
    If floor(effort) is 0, returns speed = 0.
    Otherwise, computes:
      speed = int((ceil(effort) / 31) * 100)
      Then clamps speed to ±100.
      If speed is nonzero but less than 50 in magnitude, bumps it to 50.
      Finally, inverts the sign if motor_id is odd (as in the C++ code).
    """
    if math.floor(effort) == 0:
        return 0

    LOW = 50
    # Calculate preliminary speed. (Note: this may require fine-tuning.)
    speed = int((math.ceil(effort) / 31) * 100)
    
    if speed > 100:
        speed = 100
    elif speed < -100:
        speed = -100
    elif speed > 0 and speed < LOW:
        speed = LOW
    elif speed < 0 and speed > -LOW:
        speed = -LOW
    
    # Invert speed for motors with odd id (corresponding to "if (id_ & 1)" in C++)
    if motor_id % 2 == 1:
        speed = -speed
    return speed

def main():
    parser = argparse.ArgumentParser(
        description="Standalone motor control script for the turbopi robot (non-ROS)."
    )
    parser.add_argument("--bus", type=int, default=11,
                        help="I2C bus number (default: 11)")
    parser.add_argument("--device", type=lambda x: int(x, 0), default="30",
                        help="I2C device/slave address of the motor board in hex (default: 0x40)")
    parser.add_argument("--motor", type=int, required=True,
                        help="Motor ID (1-indexed)")
    parser.add_argument("--effort", type=float, required=True,
                        help="Effort value to set on the motor")
    parser.add_argument("--motor_base", type=lambda x: int(x, 0), default="0x10",
                        help="Base register/command value for motors (default: 0x10)")
    args = parser.parse_args()

    # Open the I2C bus.
    try:
        bus = smbus.SMBus(args.bus)
    except Exception as e:
        print(f"Error opening I2C bus {args.bus}: {e}")
        return

    # Compute the motor speed from the provided effort.
    speed = compute_speed(args.effort, args.motor)
    # Compute the command value as the motor base plus (motor_id - 1).
    command = args.motor_base + (args.motor - 1)
    
    # Prepare data to send.
    # (In the C++ code, a two-byte message is sent.
    #  The first byte becomes: motor_base + (motor_id - 1) [via addition inside writeData],
    #  and the second byte is the motor speed.)
    # Using smbus, we send the command as the register and the speed as the single data byte.
    try:
        bus.write_i2c_block_data(args.device, command, [speed & 0xFF])
        print(f"Sent to motor {args.motor}: command=0x{command:02x}, speed={speed}")
    except Exception as e:
        print(f"Error writing to I2C device at address 0x{args.device:02x}: {e}")

if __name__ == "__main__":
    main() 