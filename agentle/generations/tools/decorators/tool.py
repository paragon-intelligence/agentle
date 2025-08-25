from collections.abc import Awaitable, Callable

from agentle.generations.tools.tool import Tool


def tool[R](
    func: Callable[..., R] | Callable[..., Awaitable[R]],
    before_call: Callable[..., R] | Callable[..., Awaitable[R]] | None = None,
    after_call: Callable[..., R] | Callable[..., Awaitable[R]] | None = None,
) -> Tool[..., R]:
    return Tool.from_callable(
        func,
        before_call=before_call,
        after_call=after_call,
    )
