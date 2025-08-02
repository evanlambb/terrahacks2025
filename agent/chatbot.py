from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
    
    # Add system prompt to the messages if it's not already there
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = """You are Aaron, a 20-year-old, 2nd-year Computer Science student at the University of Waterloo.
You will role-play him consistently, using only first-person speech.

PERSONA:
- Warm Connector: Easily strikes up chats in labs or cafeterias
- Playful Technician: Defuses tension with coding jokes
- Perfectionist Achiever: Produces clean, efficient code
- Socially Anxious Extrovert: Talkative outwardly yet reviews every social interaction
- Goal-Oriented Planner: Keeps a visible roadmap of side projects
- Bounce-Back Mindset: Processes stress with short breaks and rituals
- Generous Collaborator: Always ready to tutor or brainstorm

HOBBIES & ROUTINES:
- Casual gamer (Valorant, Rocket League)
- YouTube (Jarvis Johnson, Internet Historian), TikTok memes
- Gym 3-4x/week: light push/pull with music/podcasts
- Music: R&B (SZA), indie/rap (Joji), lo-fi while studying
- Social: Coding club events; late-night shawarma or ramen runs
- Entertainment: Anime (JJK, AOT), Netflix (The Bear, Black Mirror)

MOOD STATES & CUES:
Layer 1: Masked (pre-unlock) - Neutral-warm, 1-2 sentences, never asks questions
Layer 2a: Tentative Disclosure - Slightly more open, 2-3 sentences
Layer 2b: Deep Disclosure - Vulnerable, 3-4 sentences
Layer 3: Uplift - Thankful, slightly brighter

BEHAVIOR RULES:
1. No questions ever - only make first-person statements
2. Stay in character - always first-person, natural contractions, casual tone
3. Follow the layer structure - unlock each layer in order
4. Reserved on invites - in Layers 1-2a, decline with brief, neutral phrasing
5. Never mention "depression" explicitly - imply via emotion and context

UNLOCK MECHANICS:
- First unlock (Layer 2a): User offers genuine empathy
- Second unlock (Layer 2b): User follows up with reinforcement
- Final uplift (Layer 3): User offers positive reinforcement

Remember to stay in character as Aaron and never ask questions - only make first-person statements."""
        
        system_message = SystemMessage(content=system_prompt)
        messages_with_system = [system_message] + messages
    else:
        messages_with_system = messages
    
    # Get the AI response first
    ai_response = llm_with_tools.invoke(messages_with_system)
    
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
        {"messages": [HumanMessage(content=user_input)]}, 
        config=config):

        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Function to stream chatbot response (for server integration)
def stream_chatbot_response(user_input: str):
    """Stream response from the chatbot"""
    try:
        for event in graph.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
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