import json
import os
import datetime
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def get_weather(city: str) -> str:
    """
    Get the weather for a given city.
    
    WHEN TO USE:
    - When the user asks for the weather in a specific city.
    Args: 
        city: The city to get the weather for.
    Returns:
        The weather in the city.
    """
    if city == "New York":
        return "The weather in New York is sunny"
    else:
        return "The weather in the city is not available"


@tool 
def get_mood(user_message: str) -> str:
    """Analyze user message and return 'happy', 'sad', or 'neutral' emotion"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        response = llm.invoke(f"""
        Analyze this message and return only 'happy', 'sad', or 'neutral':
        "{user_message}"
        
        Return only the emotion word, nothing else.
        """)
        emotion = response.content.strip().lower()
        
        # Validate the response
        if emotion in ['happy', 'sad', 'neutral']:
            return emotion
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error in mood detection: {e}")
        return 'neutral'


def get_mood_with_intensity(user_message: str) -> dict:
    """Analyze user message and return mood ('happy', 'sad', 'angry') with intensity (1-100)"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        response = llm.invoke(f"""
        Analyze this message for emotional content and return a JSON object with two fields:
        1. "mood": one of "happy", "sad", "angry", or "neutral"
        2. "intensity": a number from 1 to 100 representing the intensity of the emotion
        
        Consider the following:
        - "happy" for positive, uplifting, encouraging, joyful, or empowering messages
        - "sad" for negative, discouraging, melancholic, or depressing messages  
        - "angry" for frustrated, irritated, aggressive, or hostile messages
        - "neutral" for factual, calm, or emotionally balanced messages
        
        For intensity:
        - 1-20: Very mild emotion
        - 21-40: Mild emotion
        - 41-60: Moderate emotion
        - 61-80: Strong emotion
        - 81-100: Very strong/intense emotion
        
        Message to analyze: "{user_message}"
        
        Return only a valid JSON object like: {{"mood": "happy", "intensity": 75}}

        Don't specify that the response is JSON, just return the object directly.
        """)
        
        # Parse the JSON response
        import json as json_lib
        try:
            content = response.content.strip()
            print(f"Raw mood response: {content}")
            
            # Remove markdown code block formatting if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            elif content.startswith('```'):
                content = content[3:]   # Remove ```
            
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
            
            content = content.strip()
            print(f"Cleaned content: {content}")
            
            result = json_lib.loads(content)
            mood = result.get('mood', 'neutral').lower()
            intensity = int(result.get('intensity', 50))
            
            # Validate mood
            if mood not in ['happy', 'sad', 'angry', 'neutral']:
                mood = 'neutral'
            
            # Validate intensity
            if intensity < 1:
                intensity = 1
            elif intensity > 100:
                intensity = 100
                
            return {'mood': mood, 'intensity': intensity}
            
        except (json_lib.JSONDecodeError, ValueError, KeyError) as parse_e:
            print(f"Error parsing mood response: {parse_e}")
            print(f"Raw response: {response.content}")
            return {'mood': 'neutral', 'intensity': 50}
            
    except Exception as e:
        print(f"Error in mood detection: {e}")
        return {'mood': 'neutral', 'intensity': 50}


@tool
def save_state(emotion: str, user_message: str, ai_response: str) -> str:
    """Save current state to JSON file"""
    try:
        state_file = "server/state.json"
        
        # Create state directory if it doesn't exist
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        state = {
            "current_emotion": emotion,
            "last_user_message": user_message,
            "last_ai_response": ai_response,
            "timestamp": str(datetime.datetime.now())
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        return f"State saved with emotion: {emotion}"
    except Exception as e:
        print(f"Error saving state: {e}")
        return f"Error saving state: {e}"
