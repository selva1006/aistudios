from langgraph.types import Command, interrupt
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from typing import Annotated


@tool
def human_assistance(query: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    human_command = Command(update={"messages": [ToolMessage(human_response["data"], tool_call_id= tool_call_id)]})
    return human_command
