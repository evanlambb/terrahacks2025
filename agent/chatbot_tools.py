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
