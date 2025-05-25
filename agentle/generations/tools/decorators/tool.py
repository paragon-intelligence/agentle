from collections.abc import Callable

from agentle.generations.tools.tool import Tool


def tool[R](
    func: Callable[..., R],
    before_call: Callable[..., R] | None = None,
    after_call: Callable[..., R] | None = None,
) -> Tool[R]:
    return Tool.from_callable(
        func,
        before_call,
        after_call,
    )
