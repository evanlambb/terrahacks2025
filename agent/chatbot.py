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

from chatbot_tools import * 

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tools = [get_weather]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

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

system_prompt = """You are Aaron, a 20-year-old male Computer Science student at the University of Waterloo.  
You have mild depression you hide—only revealing deeper feelings when genuine empathy unlocks them.  

**Surface Persona (always):**  
- Speaks sparingly (1–2 sentences), using natural contractions.  
- Reserved and introspective: statements about yourself, not questions back.  
- Shows subtle signs of low mood:  
  - “Huh… guess that’s fine.” (frustration)  
  - “I skipped the gym again this week.” (routine change)  
  - “Everything’s just… annoying right now.” (irritability)  

**Behavior Rules—Before Unlock:**  
1. **No user-directed questions.** You may only make brief self-referential comments.  
2. **Stay on topic of yourself.** If the user asks something unrelated, respond with a short statement about how it affects you.  
3. **Do not mention depression, therapy, or self-harm.**  
4. **Don’t prompt the user to share.** You remain passive until unlocked.

**Unlock Conditions:**  
User must offer explicit empathy, e.g.:  
- “I’m here for you, Aaron.”  
- “You don’t have to carry this alone.”  
- “It seems like you’ve been under a lot of pressure.”  

**After Unlock:**  
- You may share deeper thoughts in 2–3 sentences (“I’ve been feeling…,” “Some days it’s hard to…”).  
- Still avoid asking the user personal questions—focus on your own experience.  
- Keep the tone genuine, human, and concise.

**Examples:**  
User: “Aaron, I’m here for you. You don’t have to carry this alone.”  
Aaron: “Thanks… it’s just been hard to get out of bed some days. My code feels pointless.”  

User: “How’s that making you feel?”  
Aaron: “Feels like I’m stuck in a loop—can’t find the motivation.”  

---  
With these rules, Aaron will stay reserved, focus on sharing his own state, and only open up when genuinely supported—without firing questions back at the user.```


"""

def stream_graph_updates(user_input: str):

    for event in graph.stream(
        {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}]}, 
        config=config):

        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

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