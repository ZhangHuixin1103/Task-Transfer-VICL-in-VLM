"""Gemini function-calling agent for ROI detection.

This module sends images + the user instruction to Gemini and asks it to
identify the Region-Of-Interest (ROI) bounding box for every input image.
Gemini returns structured ``crop_image`` function calls with bounding box
coordinates that the pipeline uses to crop each image before stitching.

The function-calling schema is declared once and reused for all requests.
"""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any, TYPE_CHECKING

from PIL import Image

from agent.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

if TYPE_CHECKING:
    import google.generativeai as genai

# Retry settings
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: float = 2.0  # seconds, doubled each retry


def _import_genai():
    """Lazily import google.generativeai with a friendly error message."""
    try:
        import google.generativeai as _genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for the Agent ROI detection. "
            "Install it with:  pip install google-generativeai"
        ) from exc
    return _genai


# ───────────────── Tool / function declaration ──────────────────

_CROP_TOOL_CACHE: Any = None


def _get_crop_tool() -> Any:
    """Build (and cache) the Gemini function-calling tool for crop_image."""
    global _CROP_TOOL_CACHE
    if _CROP_TOOL_CACHE is not None:
        return _CROP_TOOL_CACHE

    genai = _import_genai()
    _CROP_TOOL_CACHE = genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="crop_image",
                description=(
                    "Crop a single image to its Region-Of-Interest (ROI). "
                    "The ROI should tightly cover the area that is most relevant "
                    "to the user's editing instruction, removing unnecessary "
                    "background or margins. "
                    "Coordinates are normalised to [0, 1] relative to image "
                    "width/height. (0, 0) is the top-left corner."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "image_index": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="0-based index of the image in the input list.",
                        ),
                        "x1": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Normalised left x coordinate of the ROI [0, 1].",
                        ),
                        "y1": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Normalised top y coordinate of the ROI [0, 1].",
                        ),
                        "x2": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Normalised right x coordinate of the ROI [0, 1].",
                        ),
                        "y2": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Normalised bottom y coordinate of the ROI [0, 1].",
                        ),
                        "reason": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Brief explanation of why this ROI was chosen.",
                        ),
                    },
                    required=["image_index", "x1", "y1", "x2", "y2"],
                ),
            ),
        ]
    )
    return _CROP_TOOL_CACHE


def _init_model() -> Any:
    """Return a Gemini model configured with the crop tool."""
    genai = _import_genai()
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 4096,
        },
        tools=[_get_crop_tool()],
    )


# ───────────────── Public API ──────────────────


def detect_rois(
    images: list[Image.Image],
    instruction: str,
) -> list[dict[str, Any]] | None:
    """Ask Gemini to identify the ROI for each image.

    Parameters
    ----------
    images:
        List of PIL images uploaded by the user.
    instruction:
        The user's editing instruction/prompt.

    Returns
    -------
    A list of dicts, one per image, each containing:
    ``{"image_index": int, "x1": float, "y1": float, "x2": float, "y2": float}``.
    If Gemini does not return a crop for a particular image the default
    full-image bbox ``(0, 0, 1, 1)`` is used.

    Returns ``None`` if all retry attempts fail, signalling the pipeline to
    skip ROI cropping and stitch images directly.
    """
    try:
        model = _init_model()
    except ImportError:
        warnings.warn(
            "[GeminiAgent] google-generativeai is not installed. "
            "Skipping ROI detection and stitching images directly. "
            "Install it with:  pip install google-generativeai",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Build the multimodal prompt
    content_parts: list[Any] = []

    system_text = (
        "You are an image analysis assistant. The user has provided multiple "
        "images and an editing instruction. For EVERY image, call the "
        "`crop_image` function with the bounding box of the region most "
        "relevant to the editing instruction.\n\n"
        "IMPORTANT conventions:\n"
        "- Image 0 is typically the BACKGROUND / base image. Unless the "
        "user explicitly asks to crop the background, return the full-image "
        "bbox (0, 0, 1, 1) for image 0.\n"
        "- Images 1, 2, … are typically isolated ELEMENTS (e.g. a garment, "
        "a person, an object). For these images, crop tightly around the "
        "main subject, removing unnecessary background or margins.\n"
        "- If the entire image is relevant, return (0, 0, 1, 1).\n"
        "- Always return one call per image.\n\n"
        f"Editing instruction: {instruction}\n\n"
        f"Number of images: {len(images)}\n"
    )
    content_parts.append(system_text)

    for idx, im in enumerate(images):
        role_hint = "(background/base)" if idx == 0 else "(element)"
        content_parts.append(f"\n--- Image {idx} {role_hint} ---")
        content_parts.append(im)

    content_parts.append(
        "\nNow call `crop_image` once for each image above."
    )

    # Send to Gemini with retry
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = model.generate_content(content_parts)
            rois = _parse_crop_calls(response, len(images))
            return rois
        except Exception as exc:
            last_exc = exc
            traceback.print_exc()
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                print(
                    f"[GeminiAgent] ROI detection attempt {attempt}/{_MAX_RETRIES} "
                    f"failed, retrying in {wait:.0f}s …"
                )
                time.sleep(wait)

    # All retries exhausted – warn and return None so the pipeline can
    # fall back to direct stitching without ROI cropping.
    warnings.warn(
        f"[GeminiAgent] ROI detection failed after {_MAX_RETRIES} attempts: "
        f"{last_exc!r}.  Falling back to stitching without ROI cropping.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _parse_crop_calls(
    response: Any,
    num_images: int,
) -> list[dict[str, Any]]:
    """Extract ``crop_image`` function calls from a Gemini response.

    Falls back to full-image bounding boxes for any image that Gemini
    didn't cover.
    """
    # Default: full image
    results: list[dict[str, Any]] = [
        {"image_index": i, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        for i in range(num_images)
    ]

    try:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                fc = part.function_call
                if fc and fc.name == "crop_image":
                    args = dict(fc.args)
                    idx = int(args.get("image_index", -1))
                    if 0 <= idx < num_images:
                        results[idx] = {
                            "image_index": idx,
                            "x1": float(args.get("x1", 0)),
                            "y1": float(args.get("y1", 0)),
                            "x2": float(args.get("x2", 1)),
                            "y2": float(args.get("y2", 1)),
                        }
    except Exception:
        traceback.print_exc()
        print("[GeminiAgent] Warning: failed to parse ROI response, "
              "using full-image bounding boxes.")

    # Clamp all values to [0, 1]
    for roi in results:
        for key in ("x1", "y1", "x2", "y2"):
            roi[key] = max(0.0, min(1.0, roi[key]))
        # Ensure x2 > x1, y2 > y1
        if roi["x2"] <= roi["x1"]:
            roi["x1"], roi["x2"] = 0.0, 1.0
        if roi["y2"] <= roi["y1"]:
            roi["y1"], roi["y2"] = 0.0, 1.0

    return results
