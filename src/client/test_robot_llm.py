#!/usr/bin/env python3
"""
Test script for the Robot LLM Interface.

This script demonstrates how to use the RobotLLMInterface to control a robot
using natural language commands processed by the Google Gemini model.
"""

import os
import sys
import time
from llm_interface import RobotLLMInterface

def test_robot_llm():
    """Test the Robot LLM Interface with a simple command."""
    try:
        # Create the LLM interface
        print("Initializing Robot LLM Interface...")
        llm_interface = RobotLLMInterface()
        
        # Test commands
        test_commands = [
            "Look around and tell me what you see",
            "Find a chair in the room",
            "Move forward for 2 seconds and then stop"
        ]
        
        for i, command in enumerate(test_commands):
            print(f"\nTest {i+1}: '{command}'")
            print("-" * 50)
            
            # Process the command
            response = llm_interface.process_user_query(command)
            
            print("\nFinal response:")
            print(response)
            print("-" * 50)
            
            # Wait a bit between commands
            if i < len(test_commands) - 1:
                print("Waiting 5 seconds before next command...")
                time.sleep(5)
        
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        return False

def interactive_test():
    """Run an interactive test of the Robot LLM Interface."""
    try:
        # Create the LLM interface
        print("Initializing Robot LLM Interface...")
        llm_interface = RobotLLMInterface()
        
        print("\nRobot LLM Interface initialized. Type 'exit' to quit.")
        print("Enter a task for the robot (e.g., 'find my black backpack'):")
        
        while True:
            # Get user input
            user_input = input("> ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            
            # Process the user query
            try:
                response = llm_interface.process_user_query(user_input)
                print("\nTask completed. Ready for next command.")
            except Exception as e:
                print(f"Error processing query: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"\nError during interactive test: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if the user wants to run the automated test or interactive test
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        success = interactive_test()
    else:
        success = test_robot_llm()
    
    sys.exit(0 if success else 1) 