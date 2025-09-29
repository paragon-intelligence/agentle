from enum import StrEnum


class CallEndReason(StrEnum):
    """Detailed reason why call ended"""

    # Assistant-related
    ASSISTANT_ENDED_CALL = "assistant-ended-call"
    ASSISTANT_HUNG_UP = "assistant-hung-up"
    ASSISTANT_MAX_DURATION = "assistant-max-duration"
    ASSISTANT_ERROR = "assistant-error"

    # Customer-related
    CUSTOMER_ENDED_CALL = "customer-ended-call"
    CUSTOMER_HUNG_UP = "customer-hung-up"
    CUSTOMER_DID_NOT_ANSWER = "customer-did-not-answer"
    CUSTOMER_BUSY = "customer-busy"

    # System-related
    EXCEEDED_MAX_DURATION = "exceeded-max-duration"
    PROVIDER_ERROR = "provider-error"
    NETWORK_ERROR = "network-error"
    INSUFFICIENT_FUNDS = "insufficient-funds"

    # Pipeline errors
    PIPELINE_ERROR_OPENAI_LLM_FAILED = "pipeline-error-openai-llm-failed"
    PIPELINE_ERROR_TTS_FAILED = "pipeline-error-tts-failed"
    PIPELINE_ERROR_STT_FAILED = "pipeline-error-stt-failed"
