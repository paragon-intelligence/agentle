from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class Image(BaseModel):
    """
    Represents an image extracted from a document.

    This class encapsulates the content of an image file, along with optional metadata such as
    OCR-extracted text, dimensions (height and width), original name, and alt text.

    **Attributes:**

    *   `contents` (bytes):
        The raw byte content of the image file.

        **Example:**
        ```python
        image_content = b"\\x89PNG...image data..." # Example PNG image data
        image = Image(contents=image_content)
        print(type(image.contents)) # Output: <class 'bytes'>
        ```

    *   `ocr_text` (Optional[str]):
        Text extracted from the image using Optical Character Recognition (OCR), if performed.
        Defaults to `None` if OCR was not applied or no text was found.

        **Example:**
        ```python
        image_with_ocr = Image(contents=b"...", ocr_text="Extracted text from image")
        image_no_ocr = Image(contents=b"...")
        print(image_with_ocr.ocr_text) # Output: Extracted text from image
        print(image_no_ocr.ocr_text) # Output: None
        ```

    *   `height` (Optional[float]):
        The height of the image in pixels. Defaults to `None` if the height is not available.

        **Example:**
        ```python
        image_with_dimensions = Image(contents=b"...", height=100.0, width=200.0)
        print(image_with_dimensions.height) # Output: 100.0
        ```

    *   `width` (Optional[float]):
        The width of the image in pixels. Defaults to `None` if the width is not available.

        **Example:**
        ```python
        image_with_dimensions = Image(contents=b"...", height=100.0, width=200.0)
        print(image_with_dimensions.width) # Output: 200.0
        ```

    *   `name` (Optional[str]):
        The name of the image file as it was present in the original document, if available.
        Defaults to `None`.

        **Example:**
        ```python
        named_image = Image(contents=b"...", name="logo.png")
        unnamed_image = Image(contents=b"...")
        print(named_image.name) # Output: logo.png
        print(unnamed_image.name) # Output: None
        ```

    *   `alt` (Optional[str]):
        The alternative text (alt text) associated with the image, if provided in the original document.
        Defaults to `None`.

        **Example:**
        ```python
        image_with_alt_text = Image(contents=b"...", alt="Company logo")
        image_no_alt_text = Image(contents=b"...")
        print(image_with_alt_text.alt) # Output: Company logo
        print(image_no_alt_text.alt) # Output: None
        ```

    **Usage Examples:**

    Creating a basic image with minimal information:
    ```python
    from agentle.parsing.image import Image

    # Create an image from bytes
    with open("example.png", "rb") as f:
        image_bytes = f.read()

    image = Image(contents=image_bytes)
    ```

    Creating a fully specified image with metadata:
    ```python
    from agentle.parsing.image import Image

    # Create an image with all available metadata
    detailed_image = Image(
        contents=image_bytes,
        name="chart_q2_2023.png",
        ocr_text="Revenue: $1.2M, Expenses: $0.8M, Profit: $0.4M",
        height=800.0,
        width=1200.0,
        alt="Q2 2023 Financial Performance Chart"
    )

    # This image can be included in a SectionContent object
    from agentle.parsing.section_content import SectionContent

    section = SectionContent(
        number=1,
        text="Financial Results",
        images=[detailed_image]
    )
    ```

    Accessing image data and properties:
    ```python
    # Save the image to a file
    with open("extracted_image.png", "wb") as f:
        f.write(image.contents)

    # Access metadata
    if image.ocr_text:
        print(f"Text in image: {image.ocr_text}")

    if image.width and image.height:
        aspect_ratio = image.width / image.height
        print(f"Image aspect ratio: {aspect_ratio:.2f}")
    ```
    """

    contents: bytes = Field(
        description="Contents of the image file.",
    )

    ocr_text: str | None = Field(
        default=None,
        description="Text extracted from the image using OCR.",
    )

    height: float | None = Field(
        default=None,
        description="Height of the image in pixels.",
    )

    width: float | None = Field(
        default=None,
        description="Width of the image in pixels.",
    )

    name: str | None = Field(
        default=None,
        description="The name of the image file present in the original document.",
    )

    alt: str | None = Field(
        default=None,
        description="The alt text of the image.",
    )

    model_config = ConfigDict(frozen=True)
