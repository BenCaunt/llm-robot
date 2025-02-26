import os
import cv2
import zenoh
import numpy as np
import threading
import time
import base64
from io import BytesIO
from PIL import Image

def drive_straight(power: float, duration: float) -> str:
    """
    Drive the robot forward at the given power for the given duration.
    power: normalized float [-1, 1] where +1 is forward in the camera frame. 
    duration: seconds

    Generally a safe speed is about 0.3.  If you are in a very open setting you can go up to 0.5.

    Out door open areas can go up to 1.0. 

    Due to being on real hardware it may not travel perfectly straight. 

    driving forward at 0.4 power for 1 second is approximately 1 meter.

    Returns: 
    """
    os.system(f"python actions/DriveForDuration/drive.py --power {power} --duration {duration}")

def turn_in_place(power: float, duration: float) -> str:
    """
    Turn the robot in place at the given power for the given duration.
    power: normalized float [-1, 1] where +1 is counter-clockwise (to the left) in the camera frame.
    duration: seconds

    a power of 0.4 for 0.6 seconds is approximately 90 degrees.

    Generally a safe speed is about 0.3.  If you are in a very open setting you can go up to 0.5.

    Out door open areas can go up to 1.0 though this often doesn't make a lot of sense. 
    """
    if os.system(f"python actions/DriveForDuration/turn.py --power {power} --duration {duration}") != 0:
        raise Exception("Failed to turn in place")
    return "success!"
    

def steer_to_object(object_description: str, timeout: float) -> str: 
    """
    Steer the robot to the given object.
    object_description: string description of the object we want to steer to. 

    timeout: seconds we wait until we stop.  function will return. 

    Uses a small VLM to accomplish this, usually fairly accurate.  

    Sometimes oscillates so a timeout of 5 seconds is usually a good minimum.  

    Ideally by the time this function completes the object will be in the center of the frame.  Driving forward should then move the robot towards the object.  
    """
    if os.system(f"python actions/MoondreamObjectTracking/servoing_example.py --prompt \"{object_description}\" --timeout {timeout}") != 0:
        raise Exception("Failed to steer to object")
    return "success!"

def get_current_frame():
    print("getting current frame called!!!!")
    """
    Get the current camera frame from the robot. Facing forward along the +x local frame axis.  
    
    Args:
        timeout: Maximum time to wait for a frame in seconds
        image_format: Format to encode the image as ("jpeg" or "png")
        max_size: Maximum dimension (width or height) of the returned image
    
    Returns:
        A base64-encoded string of the image that can be sent to an LLM
        
    Raises:
        TimeoutError: If no frame is received within the timeout period
    """

    timeout = 5.0
    image_format = "jpeg"
    max_size = 800

    # Initialize variables to store the frame

    latest_frame = None
    frame_received = False
    frame_lock = threading.Lock()
    
    # Callback function for the zenoh subscriber
    def on_frame(sample):
        nonlocal latest_frame, frame_received
        try:
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with frame_lock:
                latest_frame = frame
                frame_received = True
        except Exception as e:
            print(f"Error processing camera frame: {e}")
    
    # Initialize zenoh session
    session = zenoh.open(zenoh.Config())
    
    # Subscribe to the camera feed
    frame_sub = session.declare_subscriber("robot/camera/frame", on_frame)
    
    # Wait for a frame to be received
    start_time = time.time()
    while not frame_received and time.time() - start_time < timeout:
        time.sleep(0.1)
    
    # Clean up zenoh resources
    session.close()
    
    # Check if we received a frame
    if not frame_received:
        raise TimeoutError(f"No frame received within {timeout} seconds")
    
    # Process the frame for LLM consumption
    with frame_lock:
        frame = latest_frame.copy()
    
    # Convert from BGR to RGB (PIL uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image if needed
    height, width = frame_rgb.shape[:2]
    if height > max_size or width > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
    
    # Convert to PIL Image
    pil_img = Image.fromarray(frame_rgb)
    return pil_img
    
    # # Encode to base64
    # buffered = BytesIO()
    # pil_img.save(buffered, format=image_format.upper())
    # img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # return img_str

tools = [
    {
        "name": "drive_straight",
        "description": "Drive the robot forward at the given power for the given duration. Due to being on real hardware it may not travel perfectly straight.  Usecases: Moving to an object, moving to a location, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "power": {
                    "type": "number",
                    "description": "Normalized float [-1, 1] where +1 is forward in the camera frame. Generally a safe speed is about 0.3. If you are in a very open setting you can go up to 0.5. Outdoor open areas can go up to 1.0."
                },
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds to drive the robot."
                }
            },
            "required": ["power", "duration"]
        }
    },
    {
        "name": "turn_in_place",
        "description": "Turn the robot in place at the given power for the given duration.  Usecases: Turning to face an object, turning to navigate around an object, panning to find something if it isn't currently visible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "power": {
                    "type": "number",
                    "description": "Normalized float [-1, 1] where +1 is counter-clockwise (to the left) in the camera frame. Generally a safe speed is about 0.3. If you are in a very open setting you can go up to 0.5. Outdoor open areas can go up to 1.0."
                },
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds to turn the robot."
                }
            },
            "required": ["power", "duration"]
        }
    },
    {
        "name": "steer_to_object",
        "description": "Steer the robot to the given object. Uses a small VLM to accomplish this, usually fairly accurate. Sometimes oscillates so a timeout of 5 seconds is usually a good minimum. Ideally by the time this function completes the object will be in the center of the frame.  This only works if the object is currently visible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "String description of the object we want to steer to."
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds we wait until we stop and the function will return."
                }
            },
            "required": ["object_description", "timeout"]
        }
    },
    # currently not a tool because it is just added to the prompt each time. 
    # {
    #     "name": "get_current_frame",
    #     "description": "Get the current camera frame from the robot. Facing forward along the +x local frame axis. Returns a base64-encoded string of the image that can be sent to an LLM.",
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {},
    #         "required": []
    #     }
    # }
]

def generate_system_prompt(task_description: str) -> str:
    system_prompt = f"""
    You are a mobile robot tasked with doing some task in an environment.  You have a camera and can drive around with the commands provided. 
    Often times the objective will not be immediately visible, you may need to explore the enviornment to find it. Keep track of where you are and where you need to go.
    You will be responsible for object avoidance, if you expect you should have moved a large distance but haven't you might be stuck, you can backup and turn to get a different angle. 
    The turn to object tool is only useful if the object is currently visible, do not try to use it if the object is not visible. 

    You have the following tools:
    {tools}

    """
    return [system_prompt + f" your task is: {task_description}", get_current_frame()]