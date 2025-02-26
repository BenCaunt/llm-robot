#!/usr/bin/env python3
"""
Test script for Gemini API connection.
This script tests the connection to the Gemini API and verifies that the API key is working.
"""

from google import genai
import os
import sys

def test_gemini_connection():
    """Test the connection to the Gemini API"""
    try:
        # Check if the API key is set
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            print("Please set it with: export GOOGLE_API_KEY=your_api_key_here")
            return False
        
        # Initialize the client
        client = genai.Client()
        
        # Test a simple query
        print("Testing Gemini API connection...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello, can you hear me? Please respond with a short message."
        )
        
        # Print the response
        print("\nGemini response:")
        print(response.text)
        
        print("\nConnection test successful! Your Gemini API key is working.")
        return True
        
    except Exception as e:
        print(f"\nError connecting to Gemini API: {str(e)}")
        print("\nPossible issues:")
        print("1. Your API key may be invalid or expired")
        print("2. You may be experiencing rate limiting")
        print("3. There may be a network connectivity issue")
        print("4. The Gemini API service may be experiencing issues")
        return False

if __name__ == "__main__":
    success = test_gemini_connection()
    sys.exit(0 if success else 1) 