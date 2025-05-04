from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.parsing.parsed_document import ParsedDocument


def _agent_factory() -> Agent[VisualMediaDescription]: ...


class DocumentParser(BaseModel):
    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    multi_modal_provider: GenerationProvider = Field(
        default_factory=GoogleGenaiGenerationProvider,
    )
    """
    The multi-modal provider to use for generating the visual description of the document.
    Useful when you want us to customize the prompt for the visual description.
    """

    # both cannot be passed at the same time
    @model_validator(mode="before")
    def validate_both_providers_not_passed(self) -> None:
        if self.visual_description_agent and self.multi_modal_provider:
            raise ValueError(
                "Both visual_description_agent and multi_modal_provider cannot be passed at the same time"
            )

    def parse(self, document_path: str) -> ParsedDocument:
        """
        Parse a document and return a ParsedDocument.
        """
        return run_sync(self.parse_async, document_path=document_path)

    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Parse a document and return a ParsedDocument.
        """
        raise NotImplementedError("Subclasses must implement this method.")
