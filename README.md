# Mobile Robot Agent with Gemini

This project implements an autonomous mobile robot agent powered by Google Gemini that can:
- Navigate through an environment using vision
- Find and track specific objects
- Execute multi-step tasks based on natural language instructions

## Setup Requirements

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set the GOOGLE_API_KEY environment variable with your Google API key:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```

3. Ensure the robot hardware is properly connected and operational.

## Usage

1. Run the interface script:
   ```
   python src/client/llm_interface.py
   ```

2. Enter a task in natural language (e.g., "find my black backpack")

3. The robot will autonomously work to complete the task

4. You can provide follow-up instructions or start new tasks

## How It Works

- The system uses Gemini to interpret tasks and make decisions
- Gemini calls robot control functions (drive, turn, steer) as needed
- The robot continuously captures camera frames to update Gemini
- Gemini plans and executes a sequence of actions to complete the task

## Available Tools

1. **drive_straight** - Move forward/backward at a specified power for a duration
2. **turn_in_place** - Rotate left/right at a specified power for a duration
3. **steer_to_object** - Center the camera on a specified object
4. **get_current_frame** - Get the current camera view

## Example Tasks

- "Find my black backpack"
- "Navigate to the kitchen"
- "Look for a red chair"
- "Explore the room"

## Troubleshooting

If you encounter rate limiting issues with the Gemini API:
1. Check your API usage in the Google AI Studio dashboard
2. Consider upgrading your API tier if needed
3. Implement exponential backoff for retries in case of rate limiting errors 