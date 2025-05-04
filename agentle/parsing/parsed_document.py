from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.parsing.section_content import SectionContent


class ParsedDocument(BaseModel):
    name: str = Field(
        description="Name of the file",
    )

    sections: Sequence[SectionContent] = Field(
        description="Pages of the document",
    )

    @property
    def llm_described_text(self) -> str:
        sections = " ".join(
            [
                f"<section_{num}> {section.md} </section_{num}>"
                for num, section in enumerate(self.sections)
            ]
        )
        return f"<file>\n\n**name:** {self.name} \n**sections:** {sections}\n\n</file>"

    def merge_all(self, others: Sequence[ParsedDocument]) -> ParsedDocument:
        from itertools import chain

        return ParsedDocument(
            name=self.name,
            sections=list(chain(self.sections, *[other.sections for other in others])),
        )

    @classmethod
    def from_sections(
        cls, name: str, sections: Sequence[SectionContent]
    ) -> ParsedDocument:
        return cls(name=name, sections=sections)

    @classmethod
    def from_parsed_files(cls, files: Sequence[ParsedDocument]) -> ParsedDocument:
        return cls(
            name="MergedFile",
            sections=list(chain(*[file.sections for file in files])),
        )

    @property
    def md(self) -> str:
        return "\n".join([sec.md or "" for sec in self.sections])
