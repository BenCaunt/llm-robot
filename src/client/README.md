# Robot LLM Interface

This system enables controlling a robot using natural language commands processed by the Google Gemini model. The robot can perform tasks like finding objects, navigating around a space, and more based on user instructions.

## Features

- Natural language control of robot actions
- Visual perception through the robot's camera
- Self-prompting system that breaks down complex tasks
- Interactive command-line interface

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Robot with camera and movement capabilities

## Installation

1. Make sure you have the required Python packages installed:

```bash
pip install google-generativeai pillow opencv-python zenoh numpy
```

2. Set up your Google Gemini API key:
   - Create an API key using [Google AI Studio](https://ai.google.dev/)
   - Either:
     - Set the environment variable: `export GOOGLE_API_KEY=your_api_key_here`
     - Or create a file at `src/client/gemini_api_key.txt` containing your API key

## Usage

### Interactive Mode

Run the interactive test script to control the robot with natural language commands:

```bash
python src/client/test_robot_llm.py --interactive
```

This will start an interactive session where you can type commands like:
- "Find my black backpack"
- "Move to the chair in the corner"
- "Look around and tell me what you see"

### Automated Test

Run the automated test script to see a demonstration of predefined commands:

```bash
python src/client/test_robot_llm.py
```

### Direct Integration

You can also integrate the `RobotLLMInterface` class into your own Python code:

```python
from llm_interface import RobotLLMInterface

# Create the interface
llm_interface = RobotLLMInterface()

# Process a user command
response = llm_interface.process_user_query("Find my water bottle")
print(response)
```

## Available Robot Actions

The system provides the following robot control functions:

1. **drive_straight(power, duration)** - Drive the robot forward at the given power for the given duration.
   - power: normalized float [-1, 1] where +1 is forward
   - duration: seconds to drive

2. **turn_in_place(power, duration)** - Turn the robot in place at the given power for the given duration.
   - power: normalized float [-1, 1] where +1 is counter-clockwise
   - duration: seconds to turn

3. **steer_to_object(object_description, timeout)** - Steer the robot to center on the given object.
   - object_description: string description of the object
   - timeout: seconds to wait before stopping

4. **get_current_frame()** - Get the current camera frame from the robot.

## How It Works

1. The system captures the current camera frame from the robot
2. It sends the frame along with the user's command to the Gemini model
3. The model analyzes the image and plans a sequence of actions
4. The system executes the actions one by one, checking the camera between steps
5. The model provides explanations of what it's doing and why

## Troubleshooting

- If you get API key errors, make sure your Gemini API key is correctly set
- If the robot doesn't respond correctly, try more specific commands
- For movement issues, check that the robot's motors and connections are working properly

## License

This project is licensed under the MIT License - see the LICENSE file for details. 