#!/usr/bin/env python3
import zenoh
import json
import time
import argparse

def turn(duration: float, power: float):
    """
    Turn the robot for a specified duration at the given power level.
    
    Args:
        duration (float): Time to turn in seconds
        power (float): Normalized power level between -1 and 1
                      Positive values turn left, negative values turn right
    """
    # Validate power input
    if not -1 <= power <= 1:
        raise ValueError("Power must be between -1 and 1")
        
    # Initialize Zenoh session
    session = zenoh.open(zenoh.Config())
    cmd_pub = session.declare_publisher("robot/cmd")
    
    try:
        direction = "left" if power >= 0 else "right"
        print(f"Turning {direction} at {abs(power*100)}% power for {duration} seconds")
        
        # Create twist command with angular velocity
        twist_cmd = {'x': 0.0, 'theta': float(power)}
        
        # Record start time
        start_time = time.time()
        
        # Publish command until duration is reached
        while time.time() - start_time < duration:
            cmd_pub.put(json.dumps(twist_cmd))
            time.sleep(0.1)  # 10Hz update rate
            
        # Send stop command
        stop_cmd = {'x': 0.0, 'theta': 0.0}
        cmd_pub.put(json.dumps(stop_cmd))
        print("Turn complete")
        
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser(
        description="Turn the robot for a specified duration at a given power level"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        required=True,
        help="Duration to turn in seconds"
    )
    parser.add_argument(
        "--power", 
        type=float, 
        default=0.5,
        help="Power level between -1 and 1 (positive=left, negative=right, default: 0.5)"
    )
    args = parser.parse_args()
    
    try:
        turn(args.duration, args.power)
    except KeyboardInterrupt:
        print("\nTurn interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
