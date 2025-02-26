from llm_tools import drive_straight, turn_in_place, steer_to_object, get_current_frame, tools
from google import genai
import json
import time
import base64
import os
import random

"""
# Mobile Robot Agent with Gemini

This module implements an autonomous mobile robot agent powered by Google Gemini that can:
- Navigate through an environment using vision
- Find and track specific objects
- Execute multi-step tasks based on natural language instructions

## Setup Requirements:
1. Install required packages: `pip install google-generativeai`
2. Set the GOOGLE_API_KEY environment variable with your Google API key
   export GOOGLE_API_KEY=your_api_key_here
3. Ensure the robot hardware is properly connected and operational

## Usage:
1. Run this script: `python gemini_interface.py`
2. Enter a task in natural language (e.g., "find my black backpack")
3. The robot will autonomously work to complete the task
4. You can provide follow-up instructions or start new tasks

## How It Works:
- The system uses Gemini to interpret tasks and make decisions
- Gemini calls robot control functions (drive, turn, steer) as needed
- The robot continuously captures camera frames to update Gemini
- Gemini plans and executes a sequence of actions to complete the task

## Example Tasks:
- "Find my black backpack"
- "Navigate to the kitchen"
- "Look for a red chair"
- "Explore the room"
"""

# Initialize Gemini client
def initialize_client():
    """Initialize the Gemini client using the API key from environment variable"""
    try:
        # The SDK will pick up your API key from the GOOGLE_API_KEY environment variable
        client = genai.Client()
        return client
    except Exception as e:
        raise Exception(f"Error initializing Gemini client: {str(e)}")

# Convert Anthropic-style tools to Gemini-style tools
def convert_tools_to_gemini_format(tools_list):
    """Convert tools from Anthropic format to Gemini format"""
    gemini_tools = []
    
    for tool in tools_list:
        # Create a function definition for each tool
        function_def = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
        
        gemini_tools.append({"function_declarations": [function_def]})
    
    return gemini_tools

MODEL_NAME = "gemini-2.0-flash"  # Can be changed to other Gemini models

class RobotAgent:
    def __init__(self):
        self.client = initialize_client()
        self.messages = []
        self.gemini_tools = convert_tools_to_gemini_format(tools)
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
        
    def process_tool_call(self, function_call):
        """Process tool calls and execute the corresponding robot functions"""
        tool_name = function_call.name
        tool_input = json.loads(function_call.args)
        
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
            return image_b64
    
    def start_task(self, user_task):
        """Start a new task based on user instruction"""
        print(f"Starting new task: {user_task}")
        
        # Reset conversation history
        self.messages = [
            {"role": "user", "parts": [{"text": f"Task: {user_task}\n\nPlease help me complete this task. Start by getting the current camera frame to see the environment."}]}
        ]
        
        # Start the task loop
        return self.continue_task()
    
    def continue_task(self, max_iterations=10):
        """Continue the current task for a specified number of iterations"""
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\n--- Iteration {iterations}/{max_iterations} ---")
            
            # Retry mechanism with exponential backoff
            max_retries = 5
            retry_count = 0
            retry_delay = 1  # Initial delay in seconds
            
            while retry_count < max_retries:
                try:
                    # Get response from Gemini
                    response = self.client.models.generate_content(
                        model=MODEL_NAME,
                        contents=self.messages,
                        generation_config={"max_output_tokens": 4096},
                        system_instruction=self.system_prompt,
                        tools=self.gemini_tools
                    )
                    
                    # Extract text content
                    text_content = response.text
                    print(f"\nGemini's thinking:\n{text_content}")
                    
                    # Add Gemini's response to message history
                    self.messages.append({"role": "model", "parts": [{"text": text_content}]})
                    
                    # Check if Gemini wants to use a tool
                    if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                
                                # Execute the tool
                                tool_result = self.process_tool_call(function_call)
                                
                                # Add tool result to message history
                                if function_call.name == "get_current_frame":
                                    # For image results, we need to format differently
                                    self.messages.append({
                                        "role": "user", 
                                        "parts": [
                                            {"text": f"Result from {function_call.name}:"},
                                            {"inline_data": {
                                                "mime_type": "image/jpeg",
                                                "data": tool_result
                                            }}
                                        ]
                                    })
                                    # Pause briefly to let the user see what's happening
                                    time.sleep(1)
                                else:
                                    # For text results
                                    self.messages.append({
                                        "role": "user", 
                                        "parts": [{"text": f"Result from {function_call.name}: {tool_result}"}]
                                    })
                                
                                # Continue the conversation after tool use
                                return self.continue_task(max_iterations - iterations)
                    
                    # If no tool was used, Gemini has completed the task or needs user input
                    print("\nTask completed or needs user input.")
                    return text_content
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Error during Gemini API call after {max_retries} retries: {str(e)}")
                        return f"Error: {str(e)}"
                    
                    # Calculate exponential backoff with jitter
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    sleep_time = retry_delay + jitter
                    print(f"API call failed: {str(e)}. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})...")
                    time.sleep(sleep_time)
                    
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
            
        # If we've reached the maximum number of iterations
        return "Maximum number of iterations reached. Would you like to continue?"
    
    def user_input(self, user_message):
        """Handle additional user input during a task"""
        self.messages.append({"role": "user", "parts": [{"text": user_message}]})
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



