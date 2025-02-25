#!/usr/bin/env python3
import zenoh
import json
import time
import argparse

def drive_forward(duration: float, power: float):
    """
    Drive the robot for a specified duration at the given power level.
    
    Args:
        duration (float): Time to drive in seconds
        power (float): Normalized power level between -1 and 1
                      Positive values move forward, negative values move backward
    """
    # Validate power input
    if not -1 <= power <= 1:
        raise ValueError("Power must be between -1 and 1")
        
    # Initialize Zenoh session
    session = zenoh.open(zenoh.Config())
    cmd_pub = session.declare_publisher("robot/cmd")
    
    try:
        direction = "forward" if power >= 0 else "backward"
        print(f"Driving {direction} at {abs(power*100)}% power for {duration} seconds")
        
        # Create twist command with linear x velocity
        twist_cmd = {'x': float(power), 'theta': 0.0}
        
        # Record start time
        start_time = time.time()
        
        # Publish command until duration is reached
        while time.time() - start_time < duration:
            cmd_pub.put(json.dumps(twist_cmd))
            time.sleep(0.1)  # 10Hz update rate
            
        # Send stop command
        stop_cmd = {'x': 0.0, 'theta': 0.0}
        cmd_pub.put(json.dumps(stop_cmd))
        print("Drive complete")
        
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser(
        description="Drive the robot for a specified duration at a given power level"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        required=True,
        help="Duration to drive in seconds"
    )
    parser.add_argument(
        "--power", 
        type=float, 
        default=0.5,
        help="Power level between -1 and 1 (positive=forward, negative=backward, default: 0.5)"
    )
    args = parser.parse_args()
    
    try:
        drive_forward(args.duration, args.power)
    except KeyboardInterrupt:
        print("\nDrive interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
