from llm_tools import drive_straight, turn_in_place, steer_to_object, get_current_frame, tools
import anthropic
import json
import time
import base64
import os

"""
# Mobile Robot Agent with Claude

This module implements an autonomous mobile robot agent powered by Claude 3 that can:
- Navigate through an environment using vision
- Find and track specific objects
- Execute multi-step tasks based on natural language instructions

## Setup Requirements:
1. Install required packages: `pip install anthropic`
2. Create a file named 'anthropic_api_key.txt' with your Anthropic API key
3. Ensure the robot hardware is properly connected and operational

## Usage:
1. Run this script: `python claude_interface.py`
2. Enter a task in natural language (e.g., "find my black backpack")
3. The robot will autonomously work to complete the task
4. You can provide follow-up instructions or start new tasks

## How It Works:
- The system uses Claude to interpret tasks and make decisions
- Claude calls robot control functions (drive, turn, steer) as needed
- The robot continuously captures camera frames to update Claude
- Claude plans and executes a sequence of actions to complete the task

## Example Tasks:
- "Find my black backpack"
- "Navigate to the kitchen"
- "Look for a red chair"
- "Explore the room"
"""

# Load API key from file
def load_api_key(file_path="anthropic_api_key.txt"):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {file_path}. Please create this file with your Anthropic API key.")
    except Exception as e:
        raise Exception(f"Error loading API key: {str(e)}")

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=load_api_key()
)

MODEL_NAME = "claude-3-5-sonnet-20241022"  # Can be changed to other Claude models

class RobotAgent:
    def __init__(self):
        self.messages = []
        self.system_prompt = """
        You are a physical autonomous robot agent with mobility and vision capabilities. 
        Your goal is to help users complete physical tasks by navigating and interacting with the environment.
        
        You have access to the following tools:
        1. drive_straight - Move forward/backward at a specified power for a duration
        2. turn_in_place - Rotate left/right at a specified power for a duration
        3. steer_to_object - Center the camera on a specified object (only works if the object is currently visible. DO NOT USE IF YOU CANNOT SEE THE OBJECT.)
        4. get_current_frame - Get the current camera view.  Do not just assume that you can see something, really assess the image. 
        
        IMPORTANT GUIDELINES:
        - Always start by getting the current camera frame to see your environment
        - Break down complex tasks into simple steps
        - Use steer_to_object when you need to find something specific
        - Use short, careful movements (low power, short duration) when navigating
        - Continuously get new frames to update your understanding of the environment
        - Be cautious and prioritize safety - use low power (0.2-0.3) in indoor environments
        - Think step by step and explain your reasoning
        - Continue taking actions until the task is complete or impossible
        - Keep track of the steps you have taken and the steps you have not taken.  If you get stuck, you can use the steps you have not taken to help you get unstuck.
        
        For each step, you should:
        1. Analyze the current frame
        2. Decide on the next action
        3. Execute the action using the appropriate tool
        4. Get a new frame to see the result
        5. Repeat until the task is complete
        """
        
    def process_tool_call(self, tool_name, tool_input):
        """Process tool calls and execute the corresponding robot functions"""
        print(f"Executing tool: {tool_name} with input: {tool_input}")
        
        if tool_name == "drive_straight":
            result = drive_straight(tool_input["power"], tool_input["duration"])
            return "Drive command executed successfully"
            
        elif tool_name == "turn_in_place":
            result = turn_in_place(tool_input["power"], tool_input["duration"])
            return result
            
        elif tool_name == "steer_to_object":
            result = steer_to_object(tool_input["object_description"], tool_input["timeout"])
            return result
            
        elif tool_name == "get_current_frame":
            image_b64 = get_current_frame()
            return f"<image>{image_b64}</image>"
    
    def start_task(self, user_task):
        """Start a new task based on user instruction"""
        print(f"Starting new task: {user_task}")
        
        # Reset conversation history
        self.messages = [
            {"role": "user", "content": f"Task: {user_task}\n\nPlease help me complete this task. Start by getting the current camera frame to see the environment."}
        ]
        
        # Start the task loop
        return self.continue_task()
    
    def continue_task(self, max_iterations=10):
        """Continue the current task for a specified number of iterations"""
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\n--- Iteration {iterations}/{max_iterations} ---")
            
            # Get response from Claude
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=4096,
                system=self.system_prompt,  # System prompt as a separate parameter
                tools=tools,
                messages=self.messages
            )
            
            # Extract text content
            text_content = next((block.text for block in response.content if hasattr(block, "text")), "")
            print(f"\nClaude's thinking:\n{text_content}")
            
            # Add Claude's response to message history
            self.messages.append({"role": "assistant", "content": response.content})
            
            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                tool_use = next(block for block in response.content if block.type == "tool_use")
                tool_name = tool_use.name
                tool_input = tool_use.input
                
                # Execute the tool
                tool_result = self.process_tool_call(tool_name, tool_input)
                
                # Add tool result to message history
                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                })
                
                # If we got a new frame, pause briefly to let the user see what's happening
                if tool_name == "get_current_frame":
                    time.sleep(1)
            else:
                # Claude has completed the task or needs user input
                print("\nTask completed or needs user input.")
                return text_content
        
        # If we've reached the maximum number of iterations
        return "Maximum number of iterations reached. Would you like to continue?"
    
    def user_input(self, user_message):
        """Handle additional user input during a task"""
        self.messages.append({"role": "user", "content": user_message})
        return self.continue_task()

def main():
    """Main function to run the robot agent"""
    robot = RobotAgent()
    
    print("=== Robot Agent ===")
    print("Enter a task for the robot to perform (e.g., 'find my black backpack')")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nEnter task: ")
        
        if user_input.lower() == 'quit':
            break
            
        # Start the task
        result = robot.start_task(user_input)
        print(f"\nResult: {result}")
        
        # Allow for follow-up instructions
        while True:
            follow_up = input("\nFollow-up instruction (or 'new' for a new task, 'quit' to exit): ")
            
            if follow_up.lower() == 'new':
                break
            elif follow_up.lower() == 'quit':
                return
            else:
                result = robot.user_input(follow_up)
                print(f"\nResult: {result}")

if __name__ == "__main__":
    main()



