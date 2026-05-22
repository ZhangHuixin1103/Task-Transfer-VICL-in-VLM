"""Agent pipeline – orchestrates ROI detection, cropping, stitching, and recaption.

Usage example
-------------
.. code-block:: python

    from PIL import Image
    from agent import AgentPipeline

    images = [Image.open(f"img_{i}.jpg") for i in range(6)]
    instruction = "把图1中的猫放到图3的背景上，保留图2的风格"

    pipeline = AgentPipeline()
    result = pipeline.run(images, instruction)

    # result.images   – list of 2-3 composite PIL images
    # result.prompt   – rewritten instruction (~512 chars/words)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Sequence

from PIL import Image

from agent.config import MAX_OUTPUT_IMAGES, STITCH_TARGET_AREA
from agent.gemini_agent import detect_rois
from agent.image_tools import (
    build_group_mapping,
    crop_image_normalized,
    partition_and_stitch,
)
from agent.recaption import recaption


@dataclasses.dataclass
class AgentResult:
    """Container for the output of :class:`AgentPipeline`."""

    images: list[Image.Image]
    """2-3 composite images ready for FireRed-Image-Edit."""

    prompt: str
    """Rewritten instruction with updated image references."""

    group_indices: list[list[int]]
    """Mapping: ``group_indices[g]`` = list of original 0-based image indices
    that were stitched into ``images[g]``."""

    rois: list[dict]
    """Per-image ROI bounding boxes returned by Gemini."""


class AgentPipeline:
    """End-to-end pipeline that converts N (>3) images + instruction into
    2-3 composite images and a rewritten prompt suitable for
    FireRed-Image-Edit.

    When the number of input images is ≤ 3 the pipeline is a no-op and simply
    returns the images and instruction unchanged (except for recaption
    expansion).
    """

    def __init__(
        self,
        max_output_images: int = MAX_OUTPUT_IMAGES,
        target_area: int = STITCH_TARGET_AREA,
        verbose: bool = True,
    ) -> None:
        self.max_output_images = max_output_images
        self.target_area = target_area
        self.verbose = verbose

    # ────────────────────── helpers ──────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AgentPipeline] {msg}")

    # ────────────────────── main entry ──────────────────────

    def run(
        self,
        images: Sequence[Image.Image],
        instruction: str,
        enable_recaption: bool = True,
    ) -> AgentResult:
        """Run the full agent pipeline.

        Parameters
        ----------
        images:
            List of user-uploaded PIL images.
        instruction:
            The user's raw editing prompt.
        enable_recaption:
            If *True* (default), rewrite the instruction via Gemini to ~512
            words/characters.  If *False*, only update image references
            without LLM expansion.

        Returns
        -------
        :class:`AgentResult` with composite images and rewritten prompt.
        """
        images = list(images)
        n = len(images)
        self._log(f"Received {n} image(s).")

        # ------------------------------------------------------------------
        # Fast path: if ≤ 3 images, optionally recaption only
        # ------------------------------------------------------------------
        if n <= self.max_output_images:
            self._log("Image count within limit – skipping crop/stitch.")
            group_indices = [[i] for i in range(n)]
            if enable_recaption:
                prompt = recaption(instruction, group_indices)
            else:
                prompt = instruction
            return AgentResult(
                images=images,
                prompt=prompt,
                group_indices=group_indices,
                rois=[
                    {"image_index": i, "x1": 0, "y1": 0, "x2": 1, "y2": 1}
                    for i in range(n)
                ],
            )

        # ------------------------------------------------------------------
        # Step 1: ROI detection via Gemini function calling
        # ------------------------------------------------------------------
        self._log("Step 1/4  –  Detecting ROIs via Gemini …")
        t0 = time.time()
        rois = detect_rois(images, instruction)

        roi_failed = rois is None
        if roi_failed:
            self._log(
                "  ⚠ ROI detection failed – will skip cropping and "
                "stitch original images directly."
            )
            rois = [
                {"image_index": i, "x1": 0, "y1": 0, "x2": 1, "y2": 1}
                for i in range(n)
            ]
        else:
            self._log(f"  ROI detection done in {time.time() - t0:.1f}s.")
            for roi in rois:
                self._log(
                    f"  image {roi['image_index']}: "
                    f"({roi['x1']:.2f}, {roi['y1']:.2f}) – "
                    f"({roi['x2']:.2f}, {roi['y2']:.2f})"
                )

        # ------------------------------------------------------------------
        # Step 2: Crop each image to its ROI (skip if ROI detection failed)
        # ------------------------------------------------------------------
        if roi_failed:
            self._log("Step 2/4  –  Skipping crop (ROI unavailable).")
            cropped = list(images)
        else:
            self._log("Step 2/4  –  Cropping images …")
            cropped = []
            for roi in rois:
                idx = roi["image_index"]
                bbox_norm = (roi["x1"], roi["y1"], roi["x2"], roi["y2"])
                cropped_im = crop_image_normalized(images[idx], bbox_norm)
                cropped.append(cropped_im)
                self._log(
                    f"  image {idx}: {images[idx].size} → "
                    f"cropped {cropped_im.size}"
                )

        # ------------------------------------------------------------------
        # Step 3: Partition + stitch into 2-3 composite images
        #   Convention: image 0 is the background / base image and stays
        #   alone; images 1..N-1 are isolated elements stitched together.
        # ------------------------------------------------------------------
        self._log("Step 3/4  –  Stitching images …")
        group_indices, _ = build_group_mapping(
            cropped,
            max_groups=self.max_output_images,
            background_first=True,
        )
        stitched = partition_and_stitch(
            cropped,
            max_groups=self.max_output_images,
            target_area=self.target_area,
            background_first=True,
        )
        self._log(
            f"  Produced {len(stitched)} composite image(s): "
            + ", ".join(f"{im.size}" for im in stitched)
        )
        for g_idx, members in enumerate(group_indices):
            self._log(
                f"  composite {g_idx}: original images {members}"
            )

        # ------------------------------------------------------------------
        # Step 4: Recaption – rewrite the user instruction
        # ------------------------------------------------------------------
        if enable_recaption:
            self._log("Step 4/4  –  Recaptioning instruction …")
            t0 = time.time()
            new_prompt = recaption(instruction, group_indices)
            self._log(f"  Recaption done in {time.time() - t0:.1f}s.")
            self._log(f"  Original : {instruction[:120]}…")
            self._log(f"  Rewritten: {new_prompt[:120]}…")
        else:
            self._log("Step 4/4  –  Recaption disabled, updating refs only.")
            from agent.recaption import build_reference_map, _replace_image_refs
            ref_map = build_reference_map(group_indices)
            new_prompt = _replace_image_refs(instruction, ref_map)

        return AgentResult(
            images=stitched,
            prompt=new_prompt,
            group_indices=group_indices,
            rois=rois,
        )
