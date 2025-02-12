import time

"""
Small robot implementation - TurboPi

A simple wrapper for controlling the TurboPi robot's 4 motors via I2C.
"""
import smbus2
from typing import Optional

# I2C configuration
MOTOR_ADDRESS = 0x7A

# Motor IDs
FRONT_LEFT_MOTOR = 1
FRONT_RIGHT_MOTOR = 2
REAR_LEFT_MOTOR = 3
REAR_RIGHT_MOTOR = 4

class TurboPi:
    """Simple control interface for TurboPi robot"""
    
    def __init__(self, bus_number: int = 1):
        """Initialize TurboPi controller
        
        Args:
            bus_number: I2C bus number (default: 1)
        """
        self.bus = smbus2.SMBus(bus_number)
        self.address = MOTOR_ADDRESS

    def set_motor_speed(self, motor_id: int, speed: int) -> None:
        """Set the speed of a specific motor
        
        Args:
            motor_id: Motor ID (1-4)
            speed: Motor speed (-255 to 255)
        """
        if not 1 <= motor_id <= 4:
            raise ValueError("Motor ID must be between 1 and 4")
        if not -255 <= speed <= 255:
            raise ValueError("Speed must be between -255 and 255")

        # Convert to unsigned for negative values
        if speed < 0:
            speed = 256 + speed

        self.bus.write_i2c_block_data(self.address, motor_id, [speed])

    def stop_all_motors(self) -> None:
        """Stop all motors"""
        for motor_id in [FRONT_LEFT_MOTOR, FRONT_RIGHT_MOTOR, REAR_LEFT_MOTOR, REAR_RIGHT_MOTOR]:
            self.set_motor_speed(motor_id, 0)

    def drive(self, left_speed: int, right_speed: int) -> None:
        """Simple differential drive control
        
        Args:
            left_speed: Speed for left side motors (-255 to 255)
            right_speed: Speed for right side motors (-255 to 255)
        """
        # Left side
        self.set_motor_speed(FRONT_LEFT_MOTOR, left_speed)
        self.set_motor_speed(REAR_LEFT_MOTOR, left_speed)
        
        # Right side
        self.set_motor_speed(FRONT_RIGHT_MOTOR, right_speed)
        self.set_motor_speed(REAR_RIGHT_MOTOR, right_speed)

if __name__ == "__main__":
    turbo = TurboPi()
    turbo.drive(100, 100)
    time.sleep(1)
    turbo.drive(0, 0)
