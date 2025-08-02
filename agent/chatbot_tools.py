from langchain_core.tools import tool

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
