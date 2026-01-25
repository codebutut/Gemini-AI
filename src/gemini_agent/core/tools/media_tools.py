import time
import os
from pathlib import Path
from google.genai import types
from pydantic import BaseModel, Field
from . import tool, validate_args


class GenerateImageArgs(BaseModel):
    prompt: str = Field(..., description="Text description of the image to generate.")
    aspect_ratio: str | None = Field(
        "1:1", description="Aspect ratio (1:1, 4:3, 16:9)."
    )
    output_file: str | None = Field(
        None, description="Path to save the generated image."
    )


class AnalyzeImageArgs(BaseModel):
    image_path: str = Field(..., description="Path to the image file to analyze.")
    prompt: str = Field(
        "Describe this image in detail.",
        description="Question or instruction for analysis.",
    )


class CaptureScreenArgs(BaseModel):
    output_file: str | None = Field(None, description="Path to save the screenshot.")


@tool
@validate_args(GenerateImageArgs)
def generate_image(
    prompt: str, aspect_ratio: str = "1:1", output_file: str | None = None
) -> str:
    """
    Generates an image based on a text prompt using Imagen 3.

    Args:
        prompt: Text description of the image to generate.
        aspect_ratio: Aspect ratio (1:1, 4:3, 16:9).
        output_file: Path to save the generated image.

    Returns:
        str: Success message with file path or error message.
    """
    try:
        from google import genai
        from gemini_agent.config.app_config import AppConfig

        config = AppConfig()
        if not config.api_key:
            return "Error: API Key not configured."

        client = genai.Client(api_key=config.api_key)

        # Imagen 3 model
        model_id = "imagen-3.0-generate-001"

        response = client.models.generate_images(
            model=model_id,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=1,
            ),
        )

        if not response.generated_images:
            return "Error: No images were generated."

        generated_image = response.generated_images[0]
        image_data = generated_image.image.image_bytes

        if not output_file:
            output_file = f"generated_image_{int(time.time())}.png"

        with open(output_file, "wb") as f:
            f.write(image_data)
        return f"Successfully generated image and saved to '{output_file}'."
    except Exception as e:
        return f"Error generating image: {str(e)}"


@tool
@validate_args(AnalyzeImageArgs)
def analyze_image(
    image_path: str, prompt: str = "Describe this image in detail."
) -> str:
    """
    Analyzes an image file and returns a description or answers questions about it.

    Args:
        image_path: Path to the image file to analyze.
        prompt: Question or instruction for analysis.

    Returns:
        str: Analysis result or error message.
    """
    try:
        from google import genai
        from gemini_agent.config.app_config import AppConfig

        config = AppConfig()
        if not config.api_key:
            return "Error: API Key not configured."

        client = genai.Client(api_key=config.api_key)
        path = Path(image_path)

        if not path.exists():
            return f"Error: Image file '{image_path}' not found."

        # Upload file
        uploaded_file = client.files.upload(path=path)

        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = client.files.get(name=uploaded_file.name)

        response = client.models.generate_content(
            model=config.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        )

        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


@tool
@validate_args(CaptureScreenArgs)
def capture_screen(output_file: str | None = None) -> str:
    """
    Takes a screenshot of the primary monitor.

    Args:
        output_file: Path to save the screenshot.

    Returns:
        str: Success message with file path or error message.
    """
    try:
        import pyautogui

        if not output_file:
            output_file = f"screenshot_{int(time.time())}.png"

        screenshot = pyautogui.screenshot()
        screenshot.save(output_file)
        return f"Successfully captured screen and saved to '{output_file}'."
    except Exception as e:
        return f"Error capturing screen: {str(e)}"


@tool
@validate_args(AnalyzeImageArgs)
def analyze_screen(prompt: str = "Describe what is currently on the screen.") -> str:
    """
    Captures the screen and immediately analyzes it.

    Args:
        prompt: Question or instruction for analysis.

    Returns:
        str: Analysis result or error message.
    """
    try:
        output_file = f"temp_screen_{int(time.time())}.png"
        capture_result = capture_screen(output_file)

        if capture_result.startswith("Error"):
            return capture_result

        analysis_result = analyze_image(output_file, prompt)

        # Clean up temp file
        try:
            os.remove(output_file)
        except:
            pass

        return analysis_result
    except Exception as e:
        return f"Error analyzing screen: {str(e)}"
