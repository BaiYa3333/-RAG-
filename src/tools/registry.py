"""In-process registry for agent tools."""

from __future__ import annotations

from src.tools.base import BaseTool

_TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> BaseTool:
    """Register a tool instance by name and return it for decorator-style use."""

    if not tool.name:
        raise ValueError("Tool name must not be empty")
    _TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> BaseTool | None:
    return _TOOL_REGISTRY.get(name)


def list_tools() -> list[BaseTool]:
    return list(_TOOL_REGISTRY.values())


def list_tool_definitions() -> list[dict]:
    return [tool.definition() for tool in list_tools()]


async def execute_tool(name: str, arguments: dict) -> dict:
    tool = get_tool(name)
    if tool is None:
        return {"ok": False, "data": None, "error": f"Unknown tool: {name}", "metadata": {}}
    result = await tool.execute(**arguments)
    return result.to_dict()
