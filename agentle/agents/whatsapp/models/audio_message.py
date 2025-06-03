from typing import NotRequired, TypedDict


class AudioMessage(TypedDict):
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
