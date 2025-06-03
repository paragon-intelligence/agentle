from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypedDict, NotRequired

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class _Key(TypedDict):
    """Chave identificadora da mensagem WhatsApp.

    Attributes:
        remoteJid: ID do chat/contato remoto (ex: "553497722562@s.whatsapp.net")
        fromMe: Se a mensagem foi enviada por mim (True) ou recebida (False)
        id: ID único da mensagem no WhatsApp
    """

    remoteJid: str
    fromMe: bool
    id: str


class _ImageMessage(TypedDict):
    """Dados de uma mensagem com imagem do WhatsApp.

    Attributes:
        url: URL da imagem no servidor WhatsApp
        mimetype: Tipo MIME da imagem (ex: "image/jpeg")
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        height: Altura da imagem em pixels
        width: Largura da imagem em pixels
        mediaKey: Chave de criptografia para decodificar a mídia
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        jpegThumbnail: Thumbnail da imagem em base64 (opcional)
        contextInfo: Informações de contexto da mensagem (opcional)
        firstScanSidecar: Dados do primeiro scan de segurança (opcional)
        firstScanLength: Tamanho do primeiro scan (opcional)
        scansSidecar: Dados dos scans de segurança subsequentes (opcional)
        scanLengths: Lista com tamanhos dos scans realizados (opcional)
        midQualityFileSha256: Hash SHA256 da versão de qualidade média (opcional)
    """

    url: str
    mimetype: str
    fileSha256: str
    fileLength: str
    height: int
    width: int
    mediaKey: str
    fileEncSha256: str
    directPath: str
    mediaKeyTimestamp: str
    jpegThumbnail: NotRequired[str]
    contextInfo: NotRequired[dict[str, Any]]
    firstScanSidecar: NotRequired[str]
    firstScanLength: NotRequired[int]
    scansSidecar: NotRequired[str]
    scanLengths: NotRequired[list[int]]
    midQualityFileSha256: NotRequired[str]


class _DocumentMessage(TypedDict):
    """Dados de uma mensagem com documento do WhatsApp.

    Attributes:
        url: URL do documento no servidor WhatsApp
        mimetype: Tipo MIME do documento (ex: "application/pdf")
        title: Título/nome exibido do documento
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        mediaKey: Chave de criptografia para decodificar a mídia
        fileName: Nome original do arquivo
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        contactVcard: Se o documento é um cartão de contato vCard
    """

    url: str
    mimetype: str
    title: str
    fileSha256: str
    fileLength: str
    mediaKey: str
    fileName: str
    fileEncSha256: str
    directPath: str
    mediaKeyTimestamp: str
    contactVcard: bool


class _AudioMessage(TypedDict):
    """Dados de uma mensagem de áudio do WhatsApp.

    Attributes:
        url: URL do áudio no servidor WhatsApp
        mimetype: Tipo MIME do áudio (ex: "audio/ogg; codecs=opus")
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        seconds: Duração do áudio em segundos
        ptt: Se é um áudio push-to-talk (nota de voz)
        mediaKey: Chave de criptografia para decodificar a mídia
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        streamingSidecar: Dados para streaming do áudio (opcional)
        waveform: Forma de onda do áudio em base64 (opcional)
    """

    url: str
    mimetype: str
    fileSha256: str
    fileLength: str
    seconds: int
    ptt: bool
    mediaKey: str
    fileEncSha256: str
    directPath: str
    mediaKeyTimestamp: str
    streamingSidecar: NotRequired[str]
    waveform: NotRequired[str]


class _Message(TypedDict):
    """Conteúdo da mensagem WhatsApp.

    Attributes:
        conversation: Texto da mensagem (para mensagens de texto simples)
        imageMessage: Dados da imagem (para mensagens com imagem)
        documentMessage: Dados do documento (para mensagens com documento)
        audioMessage: Dados do áudio (para mensagens de áudio)
        base64: Conteúdo da mídia codificado em base64 (opcional)
    """

    conversation: NotRequired[str]
    imageMessage: NotRequired[_ImageMessage]
    documentMessage: NotRequired[_DocumentMessage]
    audioMessage: NotRequired[_AudioMessage]
    base64: NotRequired[str]


class _Data(TypedDict):
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

    key: _Key
    pushName: str
    status: str
    message: NotRequired[_Message]
    messageType: NotRequired[str]
    messageTimestamp: NotRequired[int]
    instanceId: NotRequired[str]
    source: NotRequired[str]
    contextInfo: NotRequired[dict[str, Any] | None]


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload from WhatsApp."""

    # Evolution API
    event: str | None = Field(default=None)
    instance: str | None = Field(default=None)
    data: _Data | None = Field(default=None)
    destination: str | None = Field(default=None)
    date_time: datetime | None = Field(default=None)
    sender: str | None = Field(default=None)
    server_url: str | None = Field(default=None)
    apikey: str | None = Field(default=None)

    # Meta WhatsApp Business API
    entry: list[dict[str, Any]] | None = Field(default=None)
    changes: list[dict[str, Any]] | None = Field(default=None)
    field: str | None = Field(default=None)
    value: Mapping[str, Any] | None = Field(default=None)
    phone_number_id: str | None = Field(default=None)
    metadata: Mapping[str, Any] | None = Field(default=None)
    status: str | None = Field(default=None)
    status_code: int | None = Field(default=None)
