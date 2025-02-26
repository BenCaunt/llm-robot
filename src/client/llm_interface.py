from google import genai
from google.genai import types
client = genai.Client()


from llm_tools import get_current_frame, drive_straight, turn_in_place, steer_to_object, generate_system_prompt

response = client.models.generate_content(
   model='gemini-2.0-flash',
   contents=generate_system_prompt("explore the space until you find my backpack."),
   config=types.GenerateContentConfig(
       tools=[drive_straight, turn_in_place, steer_to_object, get_current_frame],
       automatic_function_calling={'disable': True},
   ),
)

print(response.candidates[0].content.parts[0].text)
