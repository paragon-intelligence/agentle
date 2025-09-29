from enum import StrEnum


class CallType(StrEnum):
    """Type of phone call"""

    OUTBOUND = "outbound"
    INBOUND = "inbound"
    WEB = "web"
