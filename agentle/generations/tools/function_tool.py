from collections.abc import Callable
from agentle.generations.tools.tool import Tool


def function_tool(func: Callable[..., object]) -> Tool:
    return Tool.from_callable(func)
