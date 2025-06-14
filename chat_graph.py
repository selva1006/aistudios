import traceback
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

import search_tool

llm = init_chat_model("openai:gpt-4.1")

tools = [search_tool.tool]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

tools_node = ToolNode(tools)
graph_builder.add_node("tools", tools_node)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph = graph_builder.compile()

def stream_graph_updates(user_inp: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_inp}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


if __name__ == '__main__':
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye")
                break
            stream_graph_updates(user_input)
        except Exception:
            print(traceback.format_exc())
            break
