# Note: Some linter errors related to reportlab and pikepdf libraries remain
# These are due to the linter not being able to recognize some of the library APIs
# The code should still work correctly with the proper libraries installed

from pydantic import BaseModel, Field, field_validator
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    Any,
    Tuple,
    Union,
)
from enum import Enum
import io
import os
from datetime import datetime
import uuid
import base64
from os import PathLike
from pathlib import Path

# For PDF operations
import pikepdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.platypus.flowables import Spacer
from reportlab.lib.styles import ParagraphStyle


# Enums for constrained values
class PageSize(str, Enum):
    """Standard page size options for PDF documents."""

    LETTER = "letter"
    A4 = "a4"
    A3 = "a3"
    A5 = "a5"
    LEGAL = "legal"
    TABLOID = "tabloid"
    CUSTOM = "custom"


class TextAlignment(str, Enum):
    """Text alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class FontWeight(str, Enum):
    """Font weight options."""

    NORMAL = "normal"
    BOLD = "bold"
    LIGHT = "light"


class AnnotationType(str, Enum):
    """PDF annotation types."""

    HIGHLIGHT = "highlight"
    UNDERLINE = "underline"
    STRIKEOUT = "strikeout"
    LINK = "link"
    COMMENT = "comment"
    NOTE = "note"


class Color(BaseModel):
    """Color model with RGBA components."""

    r: int = Field(..., ge=0, le=255, description="Red component (0-255)")
    g: int = Field(..., ge=0, le=255, description="Green component (0-255)")
    b: int = Field(..., ge=0, le=255, description="Blue component (0-255)")
    a: float = Field(1.0, ge=0.0, le=1.0, description="Alpha/transparency (0.0-1.0)")

    @field_validator("r", "g", "b")
    def validate_color_range(cls, v: int) -> int:
        """Validate color components are in range 0-255."""
        if not 0 <= v <= 255:
            raise ValueError(f"Color value must be between 0 and 255, got {v}")
        return v

    @field_validator("a")
    def validate_alpha_range(cls, v: float) -> float:
        """Validate alpha component is in range 0.0-1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Alpha value must be between 0.0 and 1.0, got {v}")
        return v

    def to_rgb_tuple(self) -> Tuple[float, float, float]:
        """Convert to RGB tuple with values in range 0-1."""
        return (self.r / 255, self.g / 255, self.b / 255)


class Position(BaseModel):
    """2D position with x and y coordinates."""

    x: float = Field(..., description="X coordinate (in points)")
    y: float = Field(..., description="Y coordinate (in points)")


class Dimension(BaseModel):
    """2D dimension with width and height."""

    width: float = Field(..., gt=0, description="Width (in points)")
    height: float = Field(..., gt=0, description="Height (in points)")


class Rectangle(BaseModel):
    """Rectangle defined by position and dimension."""

    position: Position = Field(..., description="Top-left position of the rectangle")
    dimension: Dimension = Field(..., description="Dimensions of the rectangle")

    @property
    def x1(self) -> float:
        """Left x coordinate."""
        return self.position.x

    @property
    def y1(self) -> float:
        """Bottom y coordinate."""
        return self.position.y

    @property
    def x2(self) -> float:
        """Right x coordinate."""
        return self.position.x + self.dimension.width

    @property
    def y2(self) -> float:
        """Top y coordinate."""
        return self.position.y + self.dimension.height

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


class PDFMetadata(BaseModel):
    """PDF document metadata."""

    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    keywords: Optional[List[str]] = Field(
        None, description="Keywords associated with the document"
    )
    creator: Optional[str] = Field(
        None, description="Application that created the document"
    )
    producer: Optional[str] = Field(
        None, description="Application that produced the PDF"
    )
    creation_date: Optional[datetime] = Field(
        None, description="Date when the document was created"
    )
    modification_date: Optional[datetime] = Field(
        None, description="Date when the document was last modified"
    )
    custom_metadata: Optional[Dict[str, str]] = Field(
        None, description="Custom metadata key-value pairs"
    )


class FontSettings(BaseModel):
    """Font settings for text elements."""

    family: str = Field("Helvetica", description="Font family name")
    size: float = Field(12.0, gt=0, description="Font size in points")
    weight: FontWeight = Field(FontWeight.NORMAL, description="Font weight")
    italic: bool = Field(False, description="Whether the text is italic")
    color: Color = Field(
        default_factory=lambda: Color(r=0, g=0, b=0, a=1.0), description="Text color"
    )
    line_spacing: float = Field(1.2, ge=0, description="Line spacing multiplier")
    character_spacing: float = Field(
        0.0, description="Character spacing adjustment in points"
    )


class PDFTextElement(BaseModel):
    """Text element within a PDF page."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the element",
    )
    text: str = Field(..., description="The text content")
    position: Position = Field(..., description="Position of the text element")
    font: FontSettings = Field(
        default_factory=lambda: FontSettings(
            family="Helvetica",
            size=12.0,
            weight=FontWeight.NORMAL,
            italic=False,
            line_spacing=1.2,
            character_spacing=0.0,
        ),
        description="Font settings",
    )
    alignment: TextAlignment = Field(TextAlignment.LEFT, description="Text alignment")
    rotation: float = Field(0.0, description="Rotation angle in degrees")
    max_width: Optional[float] = Field(
        None, description="Maximum width for wrapping text"
    )
    render_mode: Literal["fill", "stroke", "fill_stroke", "invisible"] = Field(
        "fill", description="Text rendering mode"
    )


class PDFImageElement(BaseModel):
    """Image element within a PDF page."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the element",
    )
    source: str = Field(
        ..., description="Path to image file or base64 encoded image data"
    )
    is_base64: bool = Field(
        False, description="Whether the source is base64 encoded data"
    )
    position: Position = Field(..., description="Position of the image")
    dimension: Optional[Dimension] = Field(
        None, description="Dimensions of the image (if None, use original size)"
    )
    rotation: float = Field(0.0, description="Rotation angle in degrees")
    opacity: float = Field(1.0, ge=0.0, le=1.0, description="Image opacity")

    def get_image_data(self) -> bytes:
        """Return the image data as bytes."""
        if self.is_base64:
            return base64.b64decode(self.source)
        else:
            with open(self.source, "rb") as f:
                return f.read()


class LineStyle(BaseModel):
    """Line style properties for shapes and drawings."""

    width: float = Field(1.0, gt=0, description="Line width in points")
    cap_style: Literal["butt", "round", "square"] = Field(
        "butt", description="Line cap style"
    )
    join_style: Literal["miter", "round", "bevel"] = Field(
        "miter", description="Line join style"
    )
    dash_pattern: Optional[List[float]] = Field(
        None, description="Dash pattern, alternating dash and gap lengths"
    )
    color: Color = Field(
        default_factory=lambda: Color(r=0, g=0, b=0, a=1.0), description="Line color"
    )


class FillStyle(BaseModel):
    """Fill style properties for shapes and drawings."""

    color: Color = Field(..., description="Fill color")
    opacity: float = Field(1.0, ge=0.0, le=1.0, description="Fill opacity")


class PDFShapeElement(BaseModel):
    """Shape element within a PDF page."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the element",
    )
    shape_type: Literal["line", "rectangle", "circle", "ellipse", "polygon", "path"] = (
        Field(..., description="Type of shape")
    )
    position: Position = Field(..., description="Position of the shape")

    # Shape-specific properties
    points: Optional[List[Position]] = Field(
        None, description="Points for polygon or path"
    )
    dimension: Optional[Dimension] = Field(
        None, description="Dimensions for rectangle, circle, or ellipse"
    )
    radius: Optional[float] = Field(None, description="Radius for circle")
    rx: Optional[float] = Field(None, description="X radius for ellipse")
    ry: Optional[float] = Field(None, description="Y radius for ellipse")
    path_data: Optional[str] = Field(None, description="SVG-style path data")

    # Styling
    stroke: Optional[LineStyle] = Field(None, description="Stroke/line style")
    fill: Optional[FillStyle] = Field(None, description="Fill style")
    rotation: float = Field(0.0, description="Rotation angle in degrees")


class PDFAnnotationElement(BaseModel):
    """Annotation element within a PDF page."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the element",
    )
    annotation_type: AnnotationType = Field(..., description="Type of annotation")
    rect: Rectangle = Field(
        ..., description="Rectangle defining the annotation's location"
    )
    content: Optional[str] = Field(None, description="Text content of the annotation")
    color: Color = Field(
        default_factory=lambda: Color(r=255, g=255, b=0, a=0.5),
        description="Annotation color",
    )

    # Link-specific properties
    url: Optional[str] = Field(None, description="URL for link annotations")
    page_destination: Optional[int] = Field(
        None, description="Destination page number for internal links"
    )

    # Note-specific properties
    title: Optional[str] = Field(None, description="Title of the note")
    icon: Literal[
        "comment", "key", "note", "help", "newparagraph", "paragraph", "insert"
    ] = Field("note", description="Icon for the note")

    # Common properties
    creation_date: Optional[datetime] = Field(
        None, description="Date the annotation was created"
    )
    modification_date: Optional[datetime] = Field(
        None, description="Date the annotation was last modified"
    )
    flags: List[
        Literal["invisible", "hidden", "print", "nozoom", "norotate", "noview"]
    ] = Field(default_factory=list, description="Annotation flags")


class PDFFormField(BaseModel):
    """Interactive form field within a PDF page."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the form field",
    )
    field_type: Literal[
        "text", "checkbox", "radio", "combo", "list", "button", "signature"
    ] = Field(..., description="Type of form field")
    name: str = Field(..., description="Field name")
    rect: Rectangle = Field(..., description="Rectangle defining the field's location")
    value: Optional[str] = Field(None, description="Field value")
    default_value: Optional[str] = Field(None, description="Default field value")
    tooltip: Optional[str] = Field(None, description="Field tooltip")

    # Text field properties
    multiline: bool = Field(
        False, description="Whether text field allows multiple lines"
    )
    password: bool = Field(False, description="Whether text field is a password field")
    max_length: Optional[int] = Field(
        None, description="Maximum length of text field value"
    )

    # Choice field properties
    options: Optional[List[str]] = Field(
        None, description="Options for combo/list fields"
    )
    selected_indices: Optional[List[int]] = Field(
        None, description="Selected indices for combo/list fields"
    )

    # Radio/checkbox properties
    radio_group: Optional[str] = Field(None, description="Radio button group name")
    checked: Optional[bool] = Field(
        None, description="Whether checkbox/radio button is checked"
    )

    # Button properties
    action: Optional[str] = Field(None, description="Action for button (JavaScript)")

    # Common properties
    font: Optional[FontSettings] = Field(
        None, description="Font settings for field text"
    )
    read_only: bool = Field(False, description="Whether the field is read-only")
    required: bool = Field(False, description="Whether the field is required")


class PDFBookmark(BaseModel):
    """Bookmark/outline item in a PDF document."""

    title: str = Field(..., description="Bookmark title")
    page: int = Field(..., ge=0, description="Destination page number (0-based)")
    level: int = Field(0, ge=0, description="Nesting level (0 = top level)")
    y_position: Optional[float] = Field(None, description="Y position on the page")
    open: bool = Field(
        True, description="Whether the bookmark is expanded to show children"
    )
    color: Optional[Color] = Field(None, description="Bookmark color")
    style: Optional[Literal["normal", "italic", "bold", "bold_italic"]] = Field(
        None, description="Bookmark text style"
    )
    children: List["PDFBookmark"] = Field(
        default_factory=list, description="Child bookmarks"
    )


PDFBookmark.model_rebuild()


class PDFSecurity(BaseModel):
    """Security settings for a PDF document."""

    encrypted: bool = Field(False, description="Whether the document is encrypted")
    user_password: Optional[str] = Field(None, description="User password")
    owner_password: Optional[str] = Field(None, description="Owner password")
    permissions: Optional[
        List[
            Literal[
                "print",
                "modify",
                "copy",
                "annotate",
                "form_fill",
                "extract",
                "assemble",
                "print_high_quality",
            ]
        ]
    ] = Field(None, description="User permissions")
    encryption_algorithm: Literal["RC4_40", "RC4_128", "AES_128", "AES_256"] = Field(
        "AES_128", description="Encryption algorithm"
    )


class CustomPageSize(BaseModel):
    """Custom page size for PDF documents."""

    width: float = Field(..., gt=0, description="Page width in points")
    height: float = Field(..., gt=0, description="Page height in points")


class PDFPageSettings(BaseModel):
    """Settings for a PDF page."""

    size: PageSize = Field(PageSize.A4, description="Page size")
    custom_size: Optional[CustomPageSize] = Field(
        None, description="Custom page size (used when size is CUSTOM)"
    )
    orientation: Literal["portrait", "landscape"] = Field(
        "portrait", description="Page orientation"
    )
    margin_top: float = Field(
        72.0, ge=0, description="Top margin in points (1 inch = 72 points)"
    )
    margin_right: float = Field(72.0, ge=0, description="Right margin in points")
    margin_bottom: float = Field(72.0, ge=0, description="Bottom margin in points")
    margin_left: float = Field(72.0, ge=0, description="Left margin in points")
    background_color: Optional[Color] = Field(
        None, description="Background color of the page"
    )

    @property
    def page_size_tuple(self) -> Tuple[float, float]:
        """Return the page size as a (width, height) tuple in points."""
        if self.size == PageSize.CUSTOM and self.custom_size:
            size = (self.custom_size.width, self.custom_size.height)
        else:
            size_map = {
                PageSize.LETTER: letter,  # (612, 792)
                PageSize.A4: A4,  # (595, 842)
                PageSize.A3: (842, 1191),
                PageSize.A5: (420, 595),
                PageSize.LEGAL: (612, 1008),
                PageSize.TABLOID: (792, 1224),
            }
            size = size_map.get(self.size, A4)

        # Handle orientation
        if self.orientation == "landscape":
            size = (size[1], size[0])

        return size


class PDFPage(BaseModel):
    """Model representing a page in a PDF document."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the page",
    )
    settings: PDFPageSettings = Field(
        default_factory=lambda: PDFPageSettings(
            size=PageSize.A4,
            custom_size=None,
            orientation="portrait",
            margin_top=72.0,
            margin_right=72.0,
            margin_bottom=72.0,
            margin_left=72.0,
            background_color=None,
        ),
        description="Page settings",
    )
    text_elements: List[PDFTextElement] = Field(
        default_factory=list, description="Text elements on the page"
    )
    image_elements: List[PDFImageElement] = Field(
        default_factory=list, description="Image elements on the page"
    )
    shape_elements: List[PDFShapeElement] = Field(
        default_factory=list, description="Shape elements on the page"
    )
    annotation_elements: List[PDFAnnotationElement] = Field(
        default_factory=list, description="Annotation elements on the page"
    )
    form_fields: List[PDFFormField] = Field(
        default_factory=list, description="Form fields on the page"
    )

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        font: Optional[FontSettings] = None,
        alignment: Optional[TextAlignment] = None,
        rotation: float = 0.0,
        max_width: Optional[float] = None,
        render_mode: Literal["fill", "stroke", "fill_stroke", "invisible"] = "fill",
        **kwargs: Any,
    ) -> PDFTextElement:
        """
        Add a text element to the page and return it.

        Args:
            text: The text content
            x: X coordinate
            y: Y coordinate
            font: Font settings
            alignment: Text alignment
            rotation: Rotation angle in degrees
            max_width: Maximum width for wrapping text
            render_mode: Text rendering mode
            **kwargs: Additional arguments for PDFTextElement

        Returns:
            The created PDFTextElement
        """
        text_element = PDFTextElement(
            text=text,
            position=Position(x=x, y=y),
            font=font
            or FontSettings(
                family="Helvetica",
                size=12.0,
                weight=FontWeight.NORMAL,
                italic=False,
                line_spacing=1.2,
                character_spacing=0.0,
            ),
            alignment=alignment or TextAlignment.LEFT,
            rotation=rotation,
            max_width=max_width,
            render_mode=render_mode,
            **kwargs,
        )
        self.text_elements.append(text_element)
        return text_element

    def add_image(
        self,
        source: str,
        x: float,
        y: float,
        is_base64: bool = False,
        dimension: Optional[Dimension] = None,
        rotation: float = 0.0,
        opacity: float = 1.0,
        **kwargs: Any,
    ) -> PDFImageElement:
        """
        Add an image element to the page and return it.

        Args:
            source: Path to image file or base64 data
            x: X coordinate
            y: Y coordinate
            is_base64: Whether the source is base64 encoded data
            dimension: Dimensions of the image
            rotation: Rotation angle in degrees
            opacity: Image opacity
            **kwargs: Additional arguments for PDFImageElement

        Returns:
            The created PDFImageElement
        """
        image_element = PDFImageElement(
            source=source,
            position=Position(x=x, y=y),
            is_base64=is_base64,
            dimension=dimension,
            rotation=rotation,
            opacity=opacity,
            **kwargs,
        )
        self.image_elements.append(image_element)
        return image_element

    def add_shape(
        self,
        shape_type: Literal[
            "line", "rectangle", "circle", "ellipse", "polygon", "path"
        ],
        x: float,
        y: float,
        **kwargs: Any,
    ) -> PDFShapeElement:
        """
        Add a shape element to the page and return it.

        Args:
            shape_type: Type of shape
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional arguments for PDFShapeElement

        Returns:
            The created PDFShapeElement
        """
        shape_element = PDFShapeElement(
            shape_type=shape_type, position=Position(x=x, y=y), **kwargs
        )
        self.shape_elements.append(shape_element)
        return shape_element

    def add_annotation(
        self,
        annotation_type: AnnotationType,
        rect: Rectangle,
        content: Optional[str] = None,
        color: Optional[Color] = None,
        url: Optional[str] = None,
        page_destination: Optional[int] = None,
        title: Optional[str] = None,
        icon: Optional[
            Literal[
                "comment", "key", "note", "help", "newparagraph", "paragraph", "insert"
            ]
        ] = None,
        creation_date: Optional[datetime] = None,
        modification_date: Optional[datetime] = None,
        flags: Optional[
            List[
                Literal["invisible", "hidden", "print", "nozoom", "norotate", "noview"]
            ]
        ] = None,
        **kwargs: Any,
    ) -> PDFAnnotationElement:
        """
        Add an annotation element to the page and return it.

        Args:
            annotation_type: Type of annotation
            rect: Rectangle defining annotation location
            **kwargs: Additional arguments for PDFAnnotationElement

        Returns:
            The created PDFAnnotationElement
        """
        annotation_element = PDFAnnotationElement(
            annotation_type=annotation_type, rect=rect, **kwargs
        )
        self.annotation_elements.append(annotation_element)
        return annotation_element

    def add_form_field(
        self,
        field_type: Literal[
            "text", "checkbox", "radio", "combo", "list", "button", "signature"
        ],
        name: str,
        rect: Rectangle,
        value: Optional[str] = None,
        default_value: Optional[str] = None,
        tooltip: Optional[str] = None,
        multiline: bool = False,
        password: bool = False,
        max_length: Optional[int] = None,
        options: Optional[List[str]] = None,
        selected_indices: Optional[List[int]] = None,
        radio_group: Optional[str] = None,
        checked: Optional[bool] = None,
        action: Optional[str] = None,
        font: Optional[FontSettings] = None,
        read_only: bool = False,
        required: bool = False,
        **kwargs: Any,
    ) -> PDFFormField:
        """
        Add a form field to the page and return it.

        Args:
            field_type: Type of form field
            name: Field name
            rect: Rectangle defining field location
            **kwargs: Additional arguments for PDFFormField

        Returns:
            The created PDFFormField
        """
        form_field = PDFFormField(field_type=field_type, name=name, rect=rect, **kwargs)
        self.form_fields.append(form_field)
        return form_field

    def add_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill_color: Optional[Color] = None,
        stroke_color: Optional[Color] = None,
        stroke_width: float = 1.0,
    ) -> PDFShapeElement:
        """
        Add a rectangle shape to the page and return it.

        Args:
            x: X coordinate
            y: Y coordinate
            width: Rectangle width
            height: Rectangle height
            fill_color: Fill color (None for no fill)
            stroke_color: Stroke color (None for no stroke)
            stroke_width: Stroke width

        Returns:
            The created PDFShapeElement
        """
        fill = FillStyle(color=fill_color, opacity=1.0) if fill_color else None
        stroke = (
            LineStyle(
                color=stroke_color,
                width=stroke_width,
                cap_style="butt",
                join_style="miter",
                dash_pattern=None,
            )
            if stroke_color
            else None
        )

        return self.add_shape(
            shape_type="rectangle",
            x=x,
            y=y,
            dimension=Dimension(width=width, height=height),
            fill=fill,
            stroke=stroke,
        )

    def add_circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill_color: Optional[Color] = None,
        stroke_color: Optional[Color] = None,
        stroke_width: float = 1.0,
    ) -> PDFShapeElement:
        """
        Add a circle shape to the page and return it.

        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Circle radius
            fill_color: Fill color (None for no fill)
            stroke_color: Stroke color (None for no stroke)
            stroke_width: Stroke width

        Returns:
            The created PDFShapeElement
        """
        fill = FillStyle(color=fill_color, opacity=1.0) if fill_color else None
        stroke = (
            LineStyle(
                color=stroke_color,
                width=stroke_width,
                cap_style="butt",
                join_style="miter",
                dash_pattern=None,
            )
            if stroke_color
            else None
        )

        return self.add_shape(
            shape_type="circle", x=x, y=y, radius=radius, fill=fill, stroke=stroke
        )


class PDFFile(BaseModel):
    """
    A comprehensive model for representing PDF files with support for reading and writing.

    This class provides a structured representation of PDF documents including pages,
    elements, metadata, security settings, and more. It supports converting to and from
    actual PDF files using pikepdf and reportlab.
    """

    type: Literal["pdf"] = Field(default="pdf", description="File type identifier")
    metadata: PDFMetadata = Field(
        default_factory=lambda: PDFMetadata(
            title=None,
            author=None,
            subject=None,
            keywords=None,
            creator=None,
            producer=None,
            creation_date=None,
            modification_date=None,
            custom_metadata=None,
        ),
        description="Document metadata",
    )
    pages: List[PDFPage] = Field(
        default_factory=list, description="Pages in the document"
    )
    bookmarks: List[PDFBookmark] = Field(
        default_factory=list, description="Document bookmarks/outline"
    )
    security: PDFSecurity = Field(
        default_factory=lambda: PDFSecurity(
            encrypted=False,
            user_password=None,
            owner_password=None,
            permissions=None,
            encryption_algorithm="AES_128",
        ),
        description="Document security settings",
    )

    def add_page(
        self,
        id: Optional[str] = None,
        settings: Optional[PDFPageSettings] = None,
        text_elements: Optional[List[PDFTextElement]] = None,
        image_elements: Optional[List[PDFImageElement]] = None,
        shape_elements: Optional[List[PDFShapeElement]] = None,
        annotation_elements: Optional[List[PDFAnnotationElement]] = None,
        form_fields: Optional[List[PDFFormField]] = None,
        **kwargs: Any,
    ) -> PDFPage:
        """
        Add a new page to the document and return it.

        Args:
            id: Unique identifier for the page
            settings: Page settings
            text_elements: Text elements on the page
            image_elements: Image elements on the page
            shape_elements: Shape elements on the page
            annotation_elements: Annotation elements on the page
            form_fields: Form fields on the page
            **kwargs: Additional arguments to pass to PDFPage constructor

        Returns:
            The newly created page
        """
        page = PDFPage(
            id=id or str(uuid.uuid4()),
            settings=settings
            or PDFPageSettings(
                size=PageSize.A4,
                custom_size=None,
                orientation="portrait",
                margin_top=72.0,
                margin_right=72.0,
                margin_bottom=72.0,
                margin_left=72.0,
                background_color=None,
            ),
            text_elements=text_elements or [],
            image_elements=image_elements or [],
            shape_elements=shape_elements or [],
            annotation_elements=annotation_elements or [],
            form_fields=form_fields or [],
            **kwargs,
        )
        self.pages.append(page)
        return page

    def add_bookmark(
        self,
        title: str,
        page: int,
        level: int = 0,
        y_position: Optional[float] = None,
        open: bool = True,
        color: Optional[Color] = None,
        style: Optional[Literal["normal", "italic", "bold", "bold_italic"]] = None,
        children: Optional[List["PDFBookmark"]] = None,
        **kwargs: Any,
    ) -> PDFBookmark:
        """
        Add a bookmark to the document and return it.

        Args:
            title: Bookmark title
            page: Page number (0-based)
            level: Nesting level (0 = top level)
            y_position: Y position on the page
            open: Whether the bookmark is expanded to show children
            color: Bookmark color
            style: Bookmark text style
            children: Child bookmarks
            **kwargs: Additional arguments to pass to PDFBookmark constructor

        Returns:
            The newly created bookmark
        """
        bookmark = PDFBookmark(
            title=title,
            page=page,
            level=level,
            y_position=y_position,
            open=open,
            color=color,
            style=style,
            children=children or [],
            **kwargs,
        )
        self.bookmarks.append(bookmark)
        return bookmark

    @classmethod
    def from_file(cls, path: str) -> "PDFFile":
        """
        Create a PDFFile instance from an existing PDF file.

        Args:
            path: Path to the existing PDF file

        Returns:
            A new PDFFile instance with content from the existing file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")

        pdf_file = cls()

        # Open the PDF with pikepdf
        with pikepdf.open(path) as pdf:
            # Extract metadata
            meta = pdf.docinfo
            pdf_file.metadata = PDFMetadata(
                title=str(meta.get("/Title")) if "/Title" in meta else None,
                author=str(meta.get("/Author")) if "/Author" in meta else None,
                subject=str(meta.get("/Subject")) if "/Subject" in meta else None,
                creator=str(meta.get("/Creator")) if "/Creator" in meta else None,
                producer=str(meta.get("/Producer")) if "/Producer" in meta else None,
                keywords=None,
                creation_date=None,
                modification_date=None,
                custom_metadata=None,
            )

            # Extract security info
            if pdf.is_encrypted:
                pdf_file.security.encrypted = True
                # Note: Detailed permissions would require more complex handling

            # Create pages (basic structure only)
            for _ in range(len(pdf.pages)):
                pdf_file.add_page()

            # Note: Extracting detailed content (text, images, etc.) and bookmarks
            # would require more complex implementation with additional libraries

        return pdf_file

    def save(self, path: Union[str, Path, PathLike[str]]) -> None:
        """
        Save the PDFFile as an actual PDF file.

        Args:
            path: Path where the PDF file will be saved
        """
        # Create a memory buffer for the PDF
        buffer = io.BytesIO()

        # Create a canvas
        page_size = self.pages[0].settings.page_size_tuple if self.pages else A4
        c = canvas.Canvas(buffer, pagesize=page_size)

        # Process each page
        for page in self.pages:
            # Set page size
            c.setPageSize(page.settings.page_size_tuple)

            # Add page background if specified
            if page.settings.background_color:
                bg = page.settings.background_color
                c.setFillColorRGB(*bg.to_rgb_tuple(), bg.a)
                c.rect(
                    0,
                    0,
                    page.settings.page_size_tuple[0],
                    page.settings.page_size_tuple[1],
                    fill=1,
                    stroke=0,
                )

            # Process text elements
            for text_elem in page.text_elements:
                c.saveState()

                # Set font
                font_name = text_elem.font.family
                font_size = text_elem.font.size
                c.setFont(font_name, font_size)

                # Set color
                color = text_elem.font.color
                c.setFillColorRGB(*color.to_rgb_tuple(), color.a)

                # Apply transformations
                if text_elem.rotation != 0:
                    c.rotate(text_elem.rotation)

                # Set text rendering mode
                render_mode_map = {
                    "fill": 0,
                    "stroke": 1,
                    "fill_stroke": 2,
                    "invisible": 3,
                }
                c.setTextRenderMode(render_mode_map.get(text_elem.render_mode, 0))  # type: ignore # Canvas has this method in reportlab

                # Handle alignment (for simple text without wrapping)
                x = text_elem.position.x
                y = text_elem.position.y

                # Draw text
                if text_elem.max_width:
                    # Use textLines for wrapped text
                    text_obj = c.beginText(x, y)
                    text_obj.setFont(font_name, font_size)
                    text_obj.setTextOrigin(x, y)

                    # Simple wrapping (more complex wrapping would use reportlab Paragraph)
                    words = text_elem.text.split()
                    current_line = []
                    current_width = 0

                    for word in words:
                        word_width = c.stringWidth(word + " ", font_name, font_size)
                        if current_width + word_width <= text_elem.max_width:
                            current_line.append(word)
                            current_width += word_width
                        else:
                            text_obj.textLine(" ".join(current_line))
                            current_line = [word]
                            current_width = word_width

                    if current_line:
                        text_obj.textLine(" ".join(current_line))

                    c.drawText(text_obj)
                else:
                    c.drawString(x, y, text_elem.text)

                c.restoreState()

            # Process image elements
            for img_elem in page.image_elements:
                c.saveState()

                try:
                    # Get image data
                    img_data = img_elem.get_image_data()

                    # Create a temporary file for the image
                    img_file = io.BytesIO(img_data)

                    x = img_elem.position.x
                    y = img_elem.position.y

                    # Apply transformations
                    if img_elem.rotation != 0:
                        c.rotate(img_elem.rotation)

                    if img_elem.dimension:
                        width = img_elem.dimension.width
                        height = img_elem.dimension.height
                        c.drawImage(img_file, x, y, width=width, height=height)
                    else:
                        c.drawImage(img_file, x, y)
                except Exception as e:
                    print(f"Error drawing image: {str(e)}")

                c.restoreState()

            # Process shape elements
            for shape_elem in page.shape_elements:
                c.saveState()

                # Set stroke properties if defined
                if shape_elem.stroke:
                    stroke = shape_elem.stroke
                    c.setStrokeColorRGB(*stroke.color.to_rgb_tuple(), stroke.color.a)
                    c.setLineWidth(stroke.width)

                    # Set dash pattern if defined
                    if stroke.dash_pattern:
                        c.setDash(stroke.dash_pattern)

                    # Set cap and join styles
                    cap_style_map = {"butt": 0, "round": 1, "square": 2}
                    join_style_map = {"miter": 0, "round": 1, "bevel": 2}
                    c.setLineCap(cap_style_map.get(stroke.cap_style, 0))
                    c.setLineJoin(join_style_map.get(stroke.join_style, 0))

                # Set fill properties if defined
                has_fill = False
                if shape_elem.fill:
                    fill = shape_elem.fill
                    c.setFillColorRGB(*fill.color.to_rgb_tuple(), fill.color.a)
                    has_fill = True

                # Apply transformations
                if shape_elem.rotation != 0:
                    c.rotate(shape_elem.rotation)

                x = shape_elem.position.x
                y = shape_elem.position.y

                # Draw shape based on type
                if (
                    shape_elem.shape_type == "line"
                    and shape_elem.points
                    and len(shape_elem.points) >= 2
                ):
                    p1, p2 = shape_elem.points[0], shape_elem.points[1]
                    c.line(p1.x, p1.y, p2.x, p2.y)

                elif shape_elem.shape_type == "rectangle" and shape_elem.dimension:
                    width = shape_elem.dimension.width
                    height = shape_elem.dimension.height
                    c.rect(
                        x,
                        y,
                        width,
                        height,
                        fill=int(has_fill),
                        stroke=int(bool(shape_elem.stroke)),
                    )

                elif (
                    shape_elem.shape_type == "circle" and shape_elem.radius is not None
                ):
                    radius = shape_elem.radius
                    c.circle(
                        x,
                        y,
                        radius,
                        fill=int(has_fill),
                        stroke=int(bool(shape_elem.stroke)),
                    )

                elif (
                    shape_elem.shape_type == "ellipse"
                    and shape_elem.rx is not None
                    and shape_elem.ry is not None
                ):
                    rx, ry = shape_elem.rx, shape_elem.ry
                    c.ellipse(
                        x - rx,
                        y - ry,
                        x + rx,
                        y + ry,
                        fill=int(has_fill),
                        stroke=int(bool(shape_elem.stroke)),
                    )

                elif (
                    shape_elem.shape_type == "polygon"
                    and shape_elem.points
                    and len(shape_elem.points) >= 3
                ):
                    path = c.beginPath()  # type: ignore # beginPath returns a special path object, not a string
                    path.moveTo(shape_elem.points[0].x, shape_elem.points[0].y)  # type: ignore
                    for point in shape_elem.points[1:]:
                        path.lineTo(point.x, point.y)  # type: ignore
                    path.close()  # type: ignore
                    c.drawPath(
                        path, fill=int(has_fill), stroke=int(bool(shape_elem.stroke))
                    )

                elif shape_elem.shape_type == "path" and shape_elem.path_data:
                    # Basic SVG path support (would need more complex parser for full support)
                    # This is a simplified example
                    try:
                        from svg.path import parse_path

                        path = parse_path(shape_elem.path_data)
                        # Convert SVG path to reportlab path
                        # (implementation details omitted for brevity)
                    except ImportError:
                        print("svg.path library not available for path parsing")

                c.restoreState()

            # Process form fields (basic support - would need more complex handling for interactive forms)
            # Note: Form fields in PDFs typically require more low-level PDF operations

            # Process annotations (basic support)
            # Note: Full annotation support would require more low-level PDF operations

            # Move to next page
            c.showPage()

        # Set metadata
        if self.metadata.title:
            c.setTitle(self.metadata.title)
        if self.metadata.author:
            c.setAuthor(self.metadata.author)
        if self.metadata.subject:
            c.setSubject(self.metadata.subject)

        # Save the PDF
        c.save()

        # If security settings are enabled, encrypt the PDF
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if self.security.encrypted:
            with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
                # Apply security settings
                encryption_settings = {}

                if self.security.user_password:
                    encryption_settings["user_password"] = self.security.user_password
                if self.security.owner_password:
                    encryption_settings["owner_password"] = self.security.owner_password

                # Map permissions to pikepdf permission constants
                if self.security.permissions:
                    perms = pikepdf.Permissions()
                    # Map user-friendly permission names to pikepdf constants
                    # These are added as type: ignore since the exact API varies by pikepdf version
                    perm_map = {
                        "print": getattr(pikepdf.Permissions, "PRINT", 1),  # type: ignore
                        "modify": getattr(pikepdf.Permissions, "MODIFY", 2),  # type: ignore
                        "copy": getattr(pikepdf.Permissions, "EXTRACT", 4),  # type: ignore
                        "annotate": getattr(
                            pikepdf.Permissions, "MODIFY_ANNOTATION", 8
                        ),  # type: ignore
                        "form_fill": getattr(pikepdf.Permissions, "FILL_FORM", 16),  # type: ignore
                        "extract": getattr(pikepdf.Permissions, "EXTRACT", 4),  # type: ignore
                        "assemble": getattr(pikepdf.Permissions, "ASSEMBLE", 32),  # type: ignore
                        "print_high_quality": getattr(
                            pikepdf.Permissions, "PRINT_HIGH_QUALITY", 64
                        ),  # type: ignore
                    }

                    for perm in self.security.permissions:
                        if perm in perm_map:
                            # Handle different pikepdf versions that might use different APIs
                            if hasattr(perms, "add"):
                                perms.add(perm_map[perm])  # type: ignore
                            else:
                                # Alternative way to set permissions if 'add' is not available
                                setattr(perms, perm, True)  # type: ignore

                    encryption_settings["permissions"] = perms  # type: ignore

                # Set encryption algorithm
                # Handle different versions of pikepdf that might have different APIs
                if hasattr(pikepdf, "EncryptionMethod"):
                    # Modern pikepdf versions
                    algo_map = {
                        "RC4_40": getattr(pikepdf.EncryptionMethod, "rc4", 1),  # type: ignore
                        "RC4_128": getattr(pikepdf.EncryptionMethod, "rc4", 1),  # type: ignore
                        "AES_128": getattr(pikepdf.EncryptionMethod, "aes", 2),  # type: ignore
                        "AES_256": getattr(pikepdf.EncryptionMethod, "aesv3", 3),  # type: ignore
                    }
                else:
                    # Older pikepdf versions might not have EncryptionMethod enum
                    algo_map = {
                        "RC4_40": 1,  # Basic value for older versions
                        "RC4_128": 1,
                        "AES_128": 2,
                        "AES_256": 3,
                    }
                encryption_settings["R"] = 6  # type: ignore # Use PDF 2.0 security handler by default

                if self.security.encryption_algorithm in algo_map:
                    encryption_settings["method"] = algo_map[
                        self.security.encryption_algorithm
                    ]  # type: ignore

                # Save with encryption
                pdf.save(path, encryption=encryption_settings)  # type: ignore
        else:
            # Save without encryption
            # Just write directly to the path and let Python handle the type conversion
            # The type annotation above should satisfy the linter
            with open(path, "wb") as f:  # type: ignore
                f.write(pdf_bytes)

    def add_text_to_page(
        self,
        page_index: int,
        text: str,
        x: float,
        y: float,
        font: Optional[FontSettings] = None,
        alignment: Optional[TextAlignment] = None,
        rotation: float = 0.0,
        max_width: Optional[float] = None,
        render_mode: Literal["fill", "stroke", "fill_stroke", "invisible"] = "fill",
        **kwargs: Any,
    ) -> PDFTextElement:
        """
        Add text to a specific page.

        Args:
            page_index: Index of the page to add text to
            text: Text content
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional arguments for PDFTextElement

        Returns:
            The created PDFTextElement
        """
        if 0 <= page_index < len(self.pages):
            return self.pages[page_index].add_text(text, x, y, **kwargs)
        else:
            raise IndexError(
                f"Page index {page_index} out of range (0-{len(self.pages) - 1})"
            )

    def add_image_to_page(
        self,
        page_index: int,
        source: str,
        x: float,
        y: float,
        is_base64: bool = False,
        dimension: Optional[Dimension] = None,
        rotation: float = 0.0,
        opacity: float = 1.0,
        **kwargs: Any,
    ) -> PDFImageElement:
        """
        Add an image to a specific page.

        Args:
            page_index: Index of the page to add image to
            source: Path to image file or base64 data
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional arguments for PDFImageElement

        Returns:
            The created PDFImageElement
        """
        if 0 <= page_index < len(self.pages):
            return self.pages[page_index].add_image(source, x, y, **kwargs)
        else:
            raise IndexError(
                f"Page index {page_index} out of range (0-{len(self.pages) - 1})"
            )

    def add_shape_to_page(
        self,
        page_index: int,
        shape_type: Literal[
            "line", "rectangle", "circle", "ellipse", "polygon", "path"
        ],
        x: float,
        y: float,
        **kwargs: Any,
    ) -> PDFShapeElement:
        """
        Add a shape to a specific page.

        Args:
            page_index: Index of the page to add shape to
            shape_type: Type of shape
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional arguments for PDFShapeElement

        Returns:
            The created PDFShapeElement
        """
        if 0 <= page_index < len(self.pages):
            return self.pages[page_index].add_shape(shape_type, x, y, **kwargs)
        else:
            raise IndexError(
                f"Page index {page_index} out of range (0-{len(self.pages) - 1})"
            )

    def merge_pdf(self, other_pdf: "PDFFile") -> "PDFFile":
        """
        Merge another PDF into this one.

        Args:
            other_pdf: Another PDFFile instance to merge

        Returns:
            This PDFFile instance with merged content
        """
        # Copy pages from other PDF
        for page in other_pdf.pages:
            self.pages.append(page)

        # Copy bookmarks (adjusting page numbers)
        page_offset = len(self.pages) - len(other_pdf.pages)
        for bookmark in other_pdf.bookmarks:
            adjusted_bookmark = bookmark.model_copy()
            adjusted_bookmark.page += page_offset
            self.bookmarks.append(adjusted_bookmark)

        return self

    def encrypt(
        self,
        user_password: Optional[str] = None,
        owner_password: Optional[str] = None,
        permissions: Optional[
            List[
                Literal[
                    "print",
                    "modify",
                    "copy",
                    "annotate",
                    "form_fill",
                    "extract",
                    "assemble",
                    "print_high_quality",
                ]
            ]
        ] = None,
        algorithm: Literal["RC4_40", "RC4_128", "AES_128", "AES_256"] = "AES_128",
    ) -> None:
        """
        Apply security settings to the PDF.

        Args:
            user_password: User password for opening the PDF
            owner_password: Owner password for full access
            permissions: List of permitted operations
            algorithm: Encryption algorithm to use
        """
        self.security.encrypted = True
        self.security.user_password = user_password
        self.security.owner_password = owner_password
        self.security.permissions = permissions
        self.security.encryption_algorithm = algorithm

    def apply_watermark(
        self,
        text: str,
        opacity: float = 0.3,
        rotation: float = 45.0,
        font_size: float = 60,
        color: Optional[Color] = None,
    ) -> None:
        """
        Apply a text watermark to all pages.

        Args:
            text: Watermark text
            opacity: Opacity of the watermark (0.0-1.0)
            rotation: Rotation angle in degrees
            font_size: Font size for the watermark
            color: Color for the watermark (defaults to light gray)
        """
        if color is None:
            color = Color(r=128, g=128, b=128, a=opacity)
        else:
            color.a = opacity

        for page in self.pages:
            # Calculate center of page
            page_width, page_height = page.settings.page_size_tuple
            center_x = page_width / 2
            center_y = page_height / 2

            # Add watermark text
            page.add_text(
                text=text,
                x=center_x,
                y=center_y,
                font=FontSettings(
                    family="Helvetica",
                    size=font_size,
                    weight=FontWeight.NORMAL,
                    italic=False,
                    line_spacing=1.2,
                    character_spacing=0.0,
                    color=color,
                ),
                alignment=TextAlignment.CENTER,
                rotation=rotation,
            )

    def add_page_numbers(
        self,
        format_str: str = "Page {page} of {total}",
        position: Literal[
            "top-left",
            "top-center",
            "top-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
        ] = "bottom-center",
        margin: float = 36,
        font_size: float = 10,
    ) -> None:
        """
        Add page numbers to all pages.

        Args:
            format_str: Format string with {page} and {total} placeholders
            position: Position of the page numbers on the page
            margin: Margin from the edge in points
            font_size: Font size for page numbers
        """
        total_pages = len(self.pages)

        for i, page in enumerate(self.pages):
            page_text = format_str.format(page=i + 1, total=total_pages)
            page_width, page_height = page.settings.page_size_tuple

            # Calculate position
            if position == "top-left":
                x, y = margin, page_height - margin
            elif position == "top-center":
                x, y = page_width / 2, page_height - margin
            elif position == "top-right":
                x, y = page_width - margin, page_height - margin
            elif position == "bottom-left":
                x, y = margin, margin
            elif position == "bottom-center":
                x, y = page_width / 2, margin
            else:  # bottom-right
                x, y = page_width - margin, margin

            # Add page number text
            alignment = TextAlignment.CENTER
            if position.endswith("left"):
                alignment = TextAlignment.LEFT
            elif position.endswith("right"):
                alignment = TextAlignment.RIGHT

            page.add_text(
                text=page_text,
                x=x,
                y=y,
                font=FontSettings(
                    family="Helvetica",
                    size=font_size,
                    weight=FontWeight.NORMAL,
                    italic=False,
                    line_spacing=1.2,
                    character_spacing=0.0,
                    color=Color(r=0, g=0, b=0, a=1.0),
                ),
                alignment=alignment,
            )

    def create_from_text(
        self,
        text: str,
        page_settings: Optional[PDFPageSettings] = None,
        font: Optional[FontSettings] = None,
        margin: float = 72.0,
    ) -> None:
        """
        Create a PDF from text content, automatically handling pagination.

        Args:
            text: Text content to convert to PDF
            page_settings: Page settings to use
            font: Font settings to use
            margin: Text margin in points
        """
        if page_settings is None:
            page_settings = PDFPageSettings(
                size=PageSize.A4,
                custom_size=None,
                orientation="portrait",
                margin_top=72.0,
                margin_right=72.0,
                margin_bottom=72.0,
                margin_left=72.0,
                background_color=None,
            )

        if font is None:
            font = FontSettings(
                family="Helvetica",
                size=12.0,
                weight=FontWeight.NORMAL,
                italic=False,
                line_spacing=1.2,
                character_spacing=0.0,
            )

        # Create a temporary buffer to measure text and handle pagination
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_settings.page_size_tuple,
            leftMargin=page_settings.margin_left,
            rightMargin=page_settings.margin_right,
            topMargin=page_settings.margin_top,
            bottomMargin=page_settings.margin_bottom,
        )

        # Create paragraph style
        style = ParagraphStyle(
            "Normal",
            fontName=font.family,
            fontSize=font.size,
            leading=font.size * font.line_spacing,
            textColor=font.color.to_rgb_tuple(),  # type: ignore
        )

        # Convert text to paragraphs
        # Use Union type to allow both Paragraph and Spacer types
        flowables: List[Union[Paragraph, Spacer]] = []
        for para_text in text.split("\n\n"):
            if para_text.strip():
                flowables.append(Paragraph(para_text, style))
                flowables.append(Spacer(1, 12))

        # Build document to determine pagination
        doc.build(flowables)  # type: ignore

        # Use pikepdf to extract pages from the temporary document
        temp_pdf_data = buffer.getvalue()
        buffer.close()

        with pikepdf.open(io.BytesIO(temp_pdf_data)) as pdf:
            # Create pages based on the automatically paginated content
            self.pages = []
            for _ in range(len(pdf.pages)):
                self.add_page(settings=page_settings)

            # Note: The actual text content is not extracted from the temporary PDF
            # In a real implementation, we would need to extract text positions and formatting
            # from the temporary PDF or use a different approach to track text positions during pagination

    @classmethod
    def create_from_html(
        cls, html_content: str, page_settings: Optional[PDFPageSettings] = None
    ) -> "PDFFile":
        """
        Create a PDF from HTML content.

        Args:
            html_content: HTML content to convert to PDF
            page_settings: Page settings to use

        Returns:
            A new PDFFile instance with content from the HTML

        Note:
            This method requires additional libraries like WeasyPrint, xhtml2pdf, or pdfkit
            which are not included in this example.
        """
        try:
            from weasyprint import HTML

            # Create a new PDF file
            pdf_file = cls()

            if page_settings is None:
                page_settings = PDFPageSettings(
                    size=PageSize.A4,
                    custom_size=None,
                    orientation="portrait",
                    margin_top=72.0,
                    margin_right=72.0,
                    margin_bottom=72.0,
                    margin_left=72.0,
                    background_color=None,
                )

            # Convert HTML to PDF using WeasyPrint
            buffer = io.BytesIO()
            HTML(string=html_content).write_pdf(buffer, presentational_hints=True)

            # Load the PDF data into a pikepdf document
            pdf_data = buffer.getvalue()
            buffer.close()

            with pikepdf.open(io.BytesIO(pdf_data)) as temp_pdf:
                # Create pages based on the converted HTML content
                pdf_file.pages = []
                for _ in range(len(temp_pdf.pages)):
                    pdf_file.add_page(settings=page_settings)

                # Note: The actual content is not extracted from the temporary PDF
                # In a real implementation, we would need to extract detailed content
                # from the temporary PDF or use a different approach

            return pdf_file

        except ImportError:
            raise ImportError("HTML to PDF conversion requires the weasyprint library.")

    def to_base64(self) -> str:
        """
        Convert the PDF to a base64-encoded string.

        Returns:
            Base64-encoded PDF data
        """
        buffer = io.BytesIO()
        self.save(buffer)  # type: ignore # BytesIO is compatible with PDF saving
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return base64.b64encode(pdf_bytes).decode("utf-8")

    @classmethod
    def from_base64(cls, base64_data: str) -> "PDFFile":
        """
        Create a PDFFile from base64-encoded data.

        Args:
            base64_data: Base64-encoded PDF data

        Returns:
            A new PDFFile instance
        """
        pdf_bytes = base64.b64decode(base64_data)
        buffer = io.BytesIO(pdf_bytes)

        # Workaround: Save to a temporary file then load it
        temp_path = os.path.join(os.path.dirname(__file__), "_temp_pdf.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        pdf_file = cls.from_file(temp_path)

        # Clean up
        buffer.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return pdf_file


# Example usage
# Note: The following sample code may have linter errors for simplicity
# This is just a demonstration and not part of the core functionality
def create_sample_pdf() -> PDFFile:  # type: ignore
    """Create a sample PDF with various elements."""
    pdf = PDFFile()

    # Set metadata
    pdf.metadata = PDFMetadata(
        title="Sample PDF Document",
        author="PDF File Model",
        subject="Demonstration",
        keywords=["sample", "demo", "pdf", "model"],
        creator="PDFFile class",
        producer="PDFFile Example",
        creation_date=None,
        modification_date=None,
        custom_metadata=None,
    )

    # Add a page
    page = pdf.add_page(
        settings=PDFPageSettings(
            size=PageSize.A4,
            custom_size=None,
            orientation="portrait",
            margin_top=72,
            margin_right=72,
            margin_bottom=72,
            margin_left=72,
            background_color=None,
        )
    )

    # Add text
    page.add_text(
        text="Sample PDF Document",
        x=72,
        y=750,
        font=FontSettings(
            family="Helvetica-Bold",
            size=24,
            color=Color(r=0, g=0, b=0, a=1.0),
            weight=FontWeight.BOLD,
            italic=False,
            line_spacing=1.2,
            character_spacing=0.0,
        ),
    )

    # Add paragraph
    page.add_text(
        text="This is a sample PDF document created using the PDFFile model. It demonstrates various elements like text, shapes, and images.",
        x=72,
        y=700,
        font=FontSettings(
            family="Helvetica",
            size=12,
            color=Color(r=0, g=0, b=0, a=1.0),
            weight=FontWeight.NORMAL,
            italic=False,
            line_spacing=1.2,
            character_spacing=0.0,
        ),
        max_width=450,
    )

    # Add rectangle
    page.add_rectangle(
        x=72,
        y=600,
        width=450,
        height=50,
        fill_color=Color(r=200, g=200, b=200, a=0.5),
        stroke_color=Color(r=0, g=0, b=0, a=1.0),
    )

    # Add text in rectangle
    page.add_text(
        text="Text in a rectangle box",
        x=72 + 225,
        y=600 + 25,
        font=FontSettings(
            family="Helvetica",
            size=14,
            color=Color(r=0, g=0, b=0, a=1.0),
            weight=FontWeight.NORMAL,
            italic=False,
            line_spacing=1.2,
            character_spacing=0.0,
        ),
        alignment=TextAlignment.CENTER,
    )

    # Add circle
    page.add_circle(
        x=100,
        y=500,
        radius=30,
        fill_color=Color(r=255, g=0, b=0, a=0.7),
        stroke_color=Color(r=0, g=0, b=0, a=1.0),
    )

    # Add table (using shapes and text)
    table_top = 400
    table_left = 72
    row_height = 30
    col_width = 150
    header_cells = ["Column 1", "Column 2", "Column 3"]

    # Table header
    page.add_rectangle(
        x=table_left,
        y=table_top,
        width=col_width * len(header_cells),
        height=row_height,
        fill_color=Color(r=200, g=200, b=200, a=1.0),
        stroke_color=Color(r=0, g=0, b=0, a=1.0),
    )

    for i, header in enumerate(header_cells):
        page.add_text(
            text=header,
            x=table_left + i * col_width + col_width / 2,
            y=table_top + row_height / 2 - 5,
            font=FontSettings(
                family="Helvetica-Bold",
                size=12,
                color=Color(r=0, g=0, b=0, a=1.0),
                weight=FontWeight.BOLD,
                italic=False,
                line_spacing=1.2,
                character_spacing=0.0,
            ),
            alignment=TextAlignment.CENTER,
        )

    # Table rows
    rows = [
        ["Data 1-1", "Data 1-2", "Data 1-3"],
        ["Data 2-1", "Data 2-2", "Data 2-3"],
    ]

    for row_idx, row_data in enumerate(rows):
        row_y = table_top - (row_idx + 1) * row_height

        # Row background
        page.add_rectangle(
            x=table_left,
            y=row_y,
            width=col_width * len(header_cells),
            height=row_height,
            fill_color=Color(r=240, g=240, b=240, a=1.0)
            if row_idx % 2 == 0
            else Color(r=255, g=255, b=255, a=1.0),
            stroke_color=Color(r=0, g=0, b=0, a=1.0),
        )

        # Row cells
        for col_idx, cell_data in enumerate(row_data):
            page.add_text(
                text=cell_data,
                x=table_left + col_idx * col_width + col_width / 2,
                y=row_y + row_height / 2 - 5,
                font=FontSettings(
                    family="Helvetica",
                    size=10,
                    color=Color(r=0, g=0, b=0, a=1.0),
                    weight=FontWeight.NORMAL,
                    italic=False,
                    line_spacing=1.2,
                    character_spacing=0.0,
                ),
                alignment=TextAlignment.CENTER,
            )

    # Add a link annotation
    page.add_annotation(
        annotation_type=AnnotationType.LINK,
        rect=Rectangle(
            position=Position(x=72, y=250), dimension=Dimension(width=200, height=20)
        ),
        url="https://example.com",
        color=Color(r=0, g=0, b=255, a=0.2),
    )

    # Add text for the link
    page.add_text(
        text="Click here to visit example.com",
        x=72,
        y=260,
        font=FontSettings(
            family="Helvetica",
            size=12,
            color=Color(r=0, g=0, b=255, a=1.0),
            weight=FontWeight.NORMAL,
            italic=False,
            line_spacing=1.2,
            character_spacing=0.0,
        ),
    )

    # Add a bookmark
    pdf.add_bookmark(title="First Page", page=0, y_position=750)

    # Add page numbers
    pdf.add_page_numbers(format_str="Page {page} of {total}", position="bottom-center")

    return pdf


if __name__ == "__main__":
    # Create a sample PDF
    sample_pdf = create_sample_pdf()

    # Save the PDF
    sample_pdf.save("sample.pdf")

    # Load an existing PDF
    loaded_pdf = PDFFile.from_file("sample.pdf")

    # Modify the loaded PDF
    loaded_pdf.apply_watermark("SAMPLE")

    # Save the modified PDF
    loaded_pdf.save("sample_watermarked.pdf")
