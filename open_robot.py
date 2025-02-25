#!/usr/bin/env python3

import re
import subprocess
import sys
import os

# Path to the constants file
constants_path = "src/client/constants.py"

# Check if the constants file exists
if not os.path.exists(constants_path):
    print(f"Could not find constants file at {constants_path}")
    sys.exit(1)

# Extract robot hostname from constants.py
robot_host = None
try:
    with open(constants_path, 'r') as f:
        for line in f:
            if "ROBOT_HOST_NAME" in line:
                match = re.search(r'ROBOT_HOST_NAME\s*=\s*"([^"]+)"', line)
                if match:
                    robot_host = match.group(1)
                    break
except Exception as e:
    print(f"Error reading constants file: {e}")
    sys.exit(1)

if not robot_host:
    print("Could not find ROBOT_HOST_NAME in constants file")
    sys.exit(1)

# Append .local to the hostname
robot_hostname = f"{robot_host}.local"

print(f"Pinging {robot_hostname} to get IP address...")

# Ping the robot to get its IP address
try:
    ping_output = subprocess.check_output(["ping", "-c", "1", robot_hostname], universal_newlines=True)
except subprocess.CalledProcessError:
    print(f"Could not ping {robot_hostname}")
    sys.exit(1)

# Extract the IP address from the ping output
ip_match = re.search(r'\(([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\)', ping_output)
if not ip_match:
    print(f"Could not extract IP address from ping output")
    sys.exit(1)

ip_address = ip_match.group(1)
print(f"Found IP address: {ip_address}")
print(f"Connecting to {robot_host} at {ip_address}...")

# SSH to the robot
try:
    # Use the robot_host from constants file as the SSH username
    subprocess.call(["ssh", f"{robot_host}@{ip_address}"])
except Exception as e:
    print(f"Error connecting via SSH: {e}")
    sys.exit(1) 