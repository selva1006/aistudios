import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated, List
from typing_extensions import TypedDict


# Report Generation Model
generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1500)

# Reflection Model
reflection_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000)

# Prompt for Report Generation
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a creative director for making small advertisement films. You will take client requirements and "
            "come up with details like the theme of the film, the tone of the film and format of the film, with list of"
            "deliverables to the customer for customer approval to proceed to next stage of advertisement film making."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Bind the prompt to the LLM
generate_report = generation_prompt | generation_llm

# Prompt for Reflection
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are creative manager, who will review and provide feedback to the creative director's plans "
            "for creating a advertisement film before sending it for client approval."
            "The client approval document should have the details like the theme of the film, the tone of the film "
            "and format of the film, with list of deliverables from your side to customer"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Bind the prompt to the LLM
reflect_on_report = reflection_prompt | reflection_llm

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph builder
builder = StateGraph(State)

# Generation Node
async def generation_node(state: State) -> State:
    return {"messages": [await generate_report.ainvoke(state["messages"])]}

builder.add_node("generate", generation_node)

# Reflection Node
async def reflection_node(state: State) -> State:
    # Swap message roles for reflection
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect_on_report.ainvoke(translated)
    # Treat reflection as human feedback
    return {"messages": [HumanMessage(content=res.content)]}

builder.add_node("reflect", reflection_node)

# Define edges
builder.add_edge(START, "generate")

# Conditional Edge to Determine Continuation
def should_continue(state: State):
    # Limit iterations to prevent infinite loops
    if len(state["messages"]) > 6:  # Adjust based on desired iterations
        return END
    return "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

# Compile the graph with memory checkpointing
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
topic = ("I want to make advertisement film to promote my take away only restaurant. This advertisement will be for "
         "showing in cinema halls")

async def run_agent():
    async for event in graph.astream(
        {
            "messages": [
                HumanMessage(content=topic)
            ],
        },
        config,
    ):
        if "generate" in event:
            print("=== Generated Report ===")
            print(event["generate"]["messages"][-1].content)
            print("\n")
        elif "reflect" in event:
            print("=== Reflection ===")
            print(event["reflect"]["messages"][-1].content)
            print("\n")

if __name__ == "__main__":
    asyncio.run(run_agent())