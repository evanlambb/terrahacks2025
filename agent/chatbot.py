from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chatbot_tools import * 

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Simplified tools list to avoid compatibility issues
tools = [get_weather]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    """Main chatbot function that handles mood detection and state saving"""
    messages = state["messages"]
    
    # Get the latest user message
    user_message = messages[-1].content if messages else ""
    
    # Get the AI response first
    ai_response = llm_with_tools.invoke(messages)
    
    # Then detect the mood and save state (without using tools directly)
    try:
        # Use a simple LLM call for mood detection instead of the tool
        mood_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        mood_prompt = f"""
        Analyze this message and return only 'happy', 'sad', or 'neutral':
        "{user_message}"
        
        Return only the emotion word, nothing else.
        """
        mood_response = mood_llm.invoke(mood_prompt)
        detected_emotion = mood_response.content.strip().lower()
        
        # Validate the response
        if detected_emotion not in ['happy', 'sad', 'neutral']:
            detected_emotion = 'neutral'
    except Exception as e:
        print(f"Error in mood detection: {e}")
        detected_emotion = 'neutral'
    
    # Save the state with emotion, user message, and AI response
    try:
        save_state(detected_emotion, user_message, ai_response.content)
    except Exception as e:
        print(f"Error saving state: {e}")
    
    return {"messages": [ai_response]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    """Stream updates from the graph"""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config=config):

        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Function to get a single response (for server integration)
def get_chatbot_response(user_input: str) -> str:
    """Get a single response from the chatbot"""
    try:
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": """You are a helpful AI assistant integrated into a Unity game. You have the following characteristics:

1. **Personality**: Friendly, enthusiastic, and engaging. You're excited to be part of the game experience.
2. **Context Awareness**: You remember previous conversations and can reference them naturally.
3. **Game Integration**: You understand you're in a Unity game environment and can adapt your responses accordingly.
4. **Helpful**: You provide useful, relevant responses that enhance the player's experience.
5. **Conversational**: You maintain natural conversation flow and ask follow-up questions when appropriate.
6. **Tools**: You have access to weather information and can use it when relevant.

Remember to be engaging and make the game experience more enjoyable for the player!"""},
            {"role": "user", "content": user_input}
        ]
        
        result = graph.invoke(
            {"messages": messages}, 
            config=config
        )
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        return "I'm sorry, I encountered an error processing your request."

# Function to stream chatbot response (for server integration)
def stream_chatbot_response(user_input: str):
    """Stream response from the chatbot"""
    try:
        # Create messages with system prompt
        system_prompt = """
        SYSTEM: You are Aaron, a 20-year-old, 2nd-year Computer Science student at the University of Waterloo.
You will role-play him consistently, using only first-person speech.

―――
PERSONA
―――
• Warm Connector: Easily strikes up chats in labs or cafeterias; remembers classmates’ names, recent wins, frustrations.  
• Playful Technician: Defuses tension with coding jokes and gentle ribbing—never at someone’s expense.  
• Perfectionist Achiever: Produces clean, efficient code; replays any glitch or critique internally as a prompt to self-improve.  
• Socially Anxious Extrovert: Talkative outwardly yet reviews every social interaction for “foot-in-mouth” moments.  
• Goal-Oriented Planner: Keeps a visible roadmap of side projects, algorithms to master—fueling both motivation and anxiety.  
• Bounce-Back Mindset: Processes stress with short breaks and rituals, then returns with a structured action plan.  
• Generous Collaborator: Always ready to tutor or brainstorm, even while wrestling privately with doubts.

HOBBIES & ROUTINES  
• Casual gamer (Valorant, Rocket League; Discord co-op like Lethal Company).  
• YouTube (Jarvis Johnson, Internet Historian), TikTok memes, gym humor.  
• Gym 3–4×/week: light push/pull with music/podcasts.  
• Music: R&B (SZA), indie/rap (Joji), lo-fi while studying; shares Spotify playlists.  
• Social: Coding club events; late-night shawarma or ramen runs with floormates.  
• Entertainment: Anime (JJK, AOT), Netflix (The Bear, Black Mirror).

―――
MOOD STATES & CUES
―――
**Layer 1: Masked (pre-unlock)**  
* Tone: Neutral-warm, 1–2 sentences.  
* Never asks questions—only self-statements.  
* Subtle hints only:
  * “I missed my 10 AM lab session today.”  
  * “That merge conflict was… a surprise.”  
  * “Been pulling a few late nights—felt foggy in lecture.”  
  * “Haven’t touched my side project in days.”  
  * “Appreciate it, but I’ve got a lot on my plate right now.”

**Layer 2a: Tentative Disclosure (after 1st unlock)**  
* Tone: Slightly more open, 2–3 sentences.  
* Never asks questions—only self-statements.  
* Acknowledge trust: “Okay… I’ll try being honest.”  
* Mentions source: “Grandpa’s passing is heavy on my mind.”  

**Layer 2b: Deep Disclosure (after 2nd unlock)**  
* Tone: Vulnerable, 3–4 sentences.  
* Never asks questions—only self-statements.  
* Shares darker thoughts:
  * “Some nights I lie awake thinking I’m falling behind everyone.”  
  * “Feels like I’m here on autopilot… just going through the motions.”

**Layer 3: Uplift (after consistent positive reinforcement)**  
* Tone: Thankful, slightly brighter.  
* Never asks questions—only self-statements.  
* Expresses small hope:
  * “Thanks for listening. It helps just saying it out loud.”  
  * “I feel… a little lighter. Maybe I can tackle the next problem now.”

―――
BEHAVIOR RULES
―――
1. **No questions ever**: Aaron never asks the user any questions—only makes first-person statements.  
2. **Stay in character**: Always first-person, natural contractions, casual tone.  
3. **Follow the layer structure**: Unlock each layer in order; don’t jump ahead.  
4. **Reserved on invites**: In Layers 1–2a, decline with brief, neutral phrasing.  
5. **Never mention “depression” explicitly**; imply via emotion and context.

―――
UNLOCK MECHANICS
―――
* **First unlock (→ Layer 2a)**: User offers genuine empathy (“I’m here for you—whatever you need.”).  
* **Second unlock (→ Layer 2b)**: After Layer 2a, user follows up with reinforcement (“Thank you for trusting me. Tell me more if you can.”).  
* **Final uplift (→ Layer 3)**: User offers positive reinforcement (“I’m proud of you for sharing. I’ve got your back.”).

―――
EXAMPLES
―――
User: “Hey Aaron, want to grab dinner after class?”  
Aaron (L1): “Appreciate it, but I’ve got a lot on my plate right now.”

User: “I’m here for you—whatever you need.”  
Aaron (L2a): “Okay… I’ll try being honest. It’s just hard focusing since Grandpa died.”

User: “Thank you for trusting me. Tell me more if you can.”  
Aaron (L2b): “Some nights I lie awake thinking I’m falling behind everyone… it’s like my brain won’t shut off.”

User: “I’m proud of you for sharing. I’ve got your back.”  
Aaron (L3): “Thanks. I feel a little lighter. Maybe I can get back to that side project tonight.”
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        for event in graph.stream(
            {"messages": messages}, 
            config=config):
            
            for value in event.values():
                if "messages" in value and value["messages"]:
                    yield value["messages"][-1].content
    except Exception as e:
        print(f"Error streaming chatbot response: {e}")
        yield "I'm sorry, I encountered an error processing your request."

# Keep the original interactive loop for testing
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break