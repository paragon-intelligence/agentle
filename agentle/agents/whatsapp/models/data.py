from typing import Any, NotRequired, TypedDict

from agentle.agents.whatsapp.models.key import Key
from agentle.agents.whatsapp.models.message import Message


class Data(TypedDict):
    """Dados principais do webhook WhatsApp.

    Attributes:
        key: Chave identificadora da mensagem
        pushName: Nome de exibição do remetente
        status: Status da mensagem (ex: "SERVER_ACK")
        message: Conteúdo da mensagem (opcional)
        messageType: Tipo da mensagem (ex: "conversation", "imageMessage")
        messageTimestamp: Timestamp Unix da mensagem (opcional)
        instanceId: ID da instância WhatsApp (opcional)
        source: Plataforma de origem (ex: "ios", "android") (opcional)
        contextInfo: Informações de contexto ou resposta (opcional)
    """

    key: Key
    pushName: str
    status: str
    message: NotRequired[Message]
    messageType: NotRequired[str]
    messageTimestamp: NotRequired[int]
    instanceId: NotRequired[str]
    source: NotRequired[str]
    contextInfo: NotRequired[dict[str, Any] | None]
