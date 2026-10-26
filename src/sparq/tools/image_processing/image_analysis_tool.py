import base64
from pathlib import Path

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from sparq.utils.helpers import load_text

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class ImageReport(BaseModel):
    """Visual evidence and interpretation returned to the calling agent."""

    summary: str
    observations: list[str] = Field(
        description="Visible objects, layout, and relationships."
    )
    visible_text: list[str] = Field(
        description="Readable text with labels, units, and context."
    )
    charts_and_tables: list[str] = Field(
        description="Panels, axes, legends, trends, and table contents."
    )
    interpretation: str = Field(
        description="Inferred meaning, distinguished from visible evidence."
    )
    answer: str = Field(description="Answer to the caller's specific question.")
    uncertainties: list[str] = Field(
        description="Unreadable, ambiguous, or missing details."
    )


def encode_image(image_path: Path) -> str:
    """Return the local image's bytes as a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: Path) -> str | None:
    """Return the supported MIME type inferred from the file extension."""
    return MIME_TYPES.get(image_path.suffix.lower())


def make_image_analysis_tool(model: BaseChatModel, image_root: Path) -> BaseTool:
    """Create an async image tool and load its system instructions from a file."""
    system_instruction = load_text(Path(__file__).parent / "prompts" / "v0.txt")
    root = image_root.resolve()
    inspector = model.with_structured_output(ImageReport, method="json_schema")

    @tool
    async def analyze_image(image_path: str, prompt: str) -> str:
        """Analyze a local static image and return a JSON report or file error.

        Args:
            image_path: Path to an image inside the configured image root.
            prompt: Question or analysis objective for the image.
        """
        try:
            path = (root / image_path).resolve()
            if not path.is_relative_to(root):
                return "Error: image path must be inside the configured image root."
            if not path.is_file():
                return (
                    f"Error: image file does not exist or is not a regular file: {path}"
                )

            mime_type = get_mime_type(path)
            if mime_type is None:
                return "Error: unsupported image format. Use a static PNG, JPEG, or WEBP image."

            base64_image_str = encode_image(path)
        except OSError as exc:
            return f"Error: could not read image file: {exc}"

        image_url = f"data:{mime_type};base64,{base64_image_str}"
        response = await inspector.ainvoke(
            [
                SystemMessage(content=system_instruction),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                ),
            ]
        )
        if not isinstance(response, ImageReport):
            return "Error: Vision model did not return an ImageReport object (the desired structure for an image interpretration in the form of a pydantic class)"
        return response.model_dump_json()

    return analyze_image
