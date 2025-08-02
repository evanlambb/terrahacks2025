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
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, 
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
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}, 
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