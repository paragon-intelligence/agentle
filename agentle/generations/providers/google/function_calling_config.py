from typing import NotRequired, TypedDict


class FunctionCallingConfig(TypedDict):
    disable: NotRequired[bool]
    maximum_remote_calls: NotRequired[int]
    ignore_call_history: NotRequired[bool]
