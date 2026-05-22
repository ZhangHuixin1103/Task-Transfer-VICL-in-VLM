"""Image operation tools for the FireRed Agent.

Provides crop, resize, and stitch utilities used by the agent pipeline to
combine N input images into 2-3 composite images suitable for
FireRed-Image-Edit (which accepts at most 3 images).
"""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image

from agent.config import (
    STITCH_MAX_ASPECT,
    STITCH_MIN_ASPECT,
    STITCH_PAD_COLOR,
    STITCH_TARGET_AREA,
)


# ───────────────────────────── basic tools ─────────────────────────────


def crop_image(
    image: Image.Image, bbox: tuple[int, int, int, int]
) -> Image.Image:
    """Crop *image* to the bounding-box ``(x1, y1, x2, y2)``.

    Coordinates are clamped to image boundaries so out-of-range values are
    safe.

    Parameters
    ----------
    image:
        Source PIL image.
    bbox:
        ``(x1, y1, x2, y2)`` in **pixel** coordinates (inclusive start,
        exclusive end – following the PIL convention).

    Returns
    -------
    Cropped PIL image.
    """
    w, h = image.size
    x1 = max(0, min(bbox[0], w))
    y1 = max(0, min(bbox[1], h))
    x2 = max(x1, min(bbox[2], w))
    y2 = max(y1, min(bbox[3], h))
    return image.crop((x1, y1, x2, y2))


def crop_image_normalized(
    image: Image.Image,
    bbox_norm: tuple[float, float, float, float],
) -> Image.Image:
    """Crop using **normalised** coordinates in ``[0, 1]``.

    Gemini often returns bounding boxes in normalised [0, 1000] or [0, 1]
    range.  This helper accepts [0, 1] floats and converts to pixels.
    """
    w, h = image.size
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    return crop_image(image, (x1, y1, x2, y2))


def resize_image(
    image: Image.Image,
    target_width: int | None = None,
    target_height: int | None = None,
    max_side: int | None = None,
) -> Image.Image:
    """Resize an image while keeping the aspect ratio.

    Exactly one of the three keyword arguments should be provided:
    * ``target_width`` – scale so that width == target_width.
    * ``target_height`` – scale so that height == target_height.
    * ``max_side`` – scale so that the longest side == max_side.
    """
    w, h = image.size

    if max_side is not None:
        scale = max_side / max(w, h)
    elif target_width is not None:
        scale = target_width / w
    elif target_height is not None:
        scale = target_height / h
    else:
        return image.copy()

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def resize_to_area(image: Image.Image, target_area: int) -> Image.Image:
    """Scale *image* so that ``width * height ≈ target_area``."""
    w, h = image.size
    current_area = w * h
    if current_area == 0:
        return image
    scale = math.sqrt(target_area / current_area)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


# ──────────────────────── stitching helpers ─────────────────────────


def _compute_stitch_layout(
    sizes: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int]:
    """Decide how to lay out images on a canvas.

    Returns ``(positions, canvas_w, canvas_h)`` where *positions* is a list of
    ``(x, y)`` offsets for each image.

    Strategy: try row-based packing with 1 row, 2 rows, etc. and pick the
    layout that wastes the least space (smallest canvas area beyond
    the sum of image areas).
    """

    n = len(sizes)
    if n == 0:
        return [], 0, 0
    if n == 1:
        return [(0, 0)], sizes[0][0], sizes[0][1]

    best_positions: list[tuple[int, int]] = []
    best_canvas = (0, 0)
    best_waste = float("inf")

    # Try splitting images into `num_rows` rows
    for num_rows in range(1, n + 1):
        rows: list[list[int]] = [[] for _ in range(num_rows)]
        # Distribute images to rows round-robin (widest first for balance)
        indices_sorted = sorted(range(n), key=lambda i: sizes[i][0], reverse=True)
        row_widths = [0] * num_rows
        for idx in indices_sorted:
            # Put into the row with the smallest current width
            min_row = min(range(num_rows), key=lambda r: row_widths[r])
            rows[min_row].append(idx)
            row_widths[min_row] += sizes[idx][0]

        # Remove empty rows
        rows = [r for r in rows if r]

        # For each row, images are placed left-to-right.
        # Row height = max image height in the row.
        positions = [None] * n
        canvas_w = 0
        y_offset = 0
        for row in rows:
            row_h = max(sizes[i][1] for i in row)
            x_offset = 0
            for i in row:
                positions[i] = (x_offset, y_offset)
                x_offset += sizes[i][0]
            canvas_w = max(canvas_w, x_offset)
            y_offset += row_h

        canvas_h = y_offset
        total_img_area = sum(w * h for w, h in sizes)
        waste = canvas_w * canvas_h - total_img_area

        aspect = canvas_w / max(canvas_h, 1)
        if aspect < STITCH_MIN_ASPECT or aspect > STITCH_MAX_ASPECT:
            waste += 1e9  # penalise extreme aspect ratios

        if waste < best_waste:
            best_waste = waste
            best_positions = positions  # type: ignore[assignment]
            best_canvas = (canvas_w, canvas_h)

    return best_positions, best_canvas[0], best_canvas[1]


def stitch_images(
    images: Sequence[Image.Image],
    target_area: int = STITCH_TARGET_AREA,
    pad_color: tuple[int, int, int] = STITCH_PAD_COLOR,
) -> Image.Image:
    """Stitch multiple images into a single composite image.

    The algorithm:
    1. Scale every image so that the **sum** of individual areas equals
       *target_area*. This keeps each sub-image as large as possible.
    2. Use a simple row-packing layout to minimise whitespace.
    3. Paste images onto a canvas and pad remaining pixels.

    Parameters
    ----------
    images:
        Input images (already cropped to ROIs).
    target_area:
        Desired total canvas area in pixels (default 1024*1024).
    pad_color:
        RGB colour for padding.

    Returns
    -------
    A single composite PIL image.
    """
    if len(images) == 0:
        raise ValueError("At least one image is required.")
    if len(images) == 1:
        return resize_to_area(images[0], target_area)

    # 1) Compute uniform scale so that sum-of-areas ≈ target_area
    total_area = sum(im.size[0] * im.size[1] for im in images)
    if total_area == 0:
        raise ValueError("All images have zero area.")
    scale = math.sqrt(target_area / total_area)

    scaled: list[Image.Image] = []
    for im in images:
        new_w = max(1, round(im.size[0] * scale))
        new_h = max(1, round(im.size[1] * scale))
        scaled.append(im.resize((new_w, new_h), Image.LANCZOS))

    sizes = [im.size for im in scaled]

    # 2) Determine layout
    positions, canvas_w, canvas_h = _compute_stitch_layout(sizes)

    # 3) Create canvas and paste
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=pad_color)
    for im, (px, py) in zip(scaled, positions):
        canvas.paste(im, (px, py))

    return canvas


def partition_and_stitch(
    images: Sequence[Image.Image],
    max_groups: int = 3,
    target_area: int = STITCH_TARGET_AREA,
    pad_color: tuple[int, int, int] = STITCH_PAD_COLOR,
    background_first: bool = False,
) -> list[Image.Image]:
    """Partition *images* into ≤ *max_groups* groups and stitch each group.

    The goal is to convert N (>3) images into 2-3 composite images.  We try to
    balance the groups by total pixel area so that each stitched output is
    roughly the same size.

    Parameters
    ----------
    images:
        All input images (already cropped to ROI if applicable).
    max_groups:
        Maximum number of output composite images (2 or 3).
    target_area:
        Target canvas area per composite image.
    pad_color:
        Padding colour.
    background_first:
        If *True*, treat ``images[0]`` as the background / base image and
        keep it as a standalone composite.  The remaining images (isolated
        elements) are stitched into 1-2 additional composites.

    Returns
    -------
    A list of 2-3 stitched composite images.
    """
    n = len(images)
    if n <= max_groups:
        # Each image can be its own group – just resize.
        return [resize_to_area(im, target_area) for im in images]

    # ── Background-first strategy ──────────────────────────────────────
    if background_first and n > max_groups:
        # Group 0 = background image alone
        bg = resize_to_area(images[0], target_area)
        rest = list(images[1:])
        rest_max_groups = max_groups - 1  # 1 or 2 groups for elements

        if len(rest) <= rest_max_groups:
            return [bg] + [resize_to_area(im, target_area) for im in rest]

        # Partition the element images
        element_groups: list[list[Image.Image]] = [
            [] for _ in range(rest_max_groups)
        ]
        element_areas: list[int] = [0] * rest_max_groups

        indexed = sorted(
            enumerate(rest),
            key=lambda t: t[1].size[0] * t[1].size[1],
            reverse=True,
        )
        for _, im in indexed:
            min_g = min(
                range(rest_max_groups), key=lambda g: element_areas[g]
            )
            element_groups[min_g].append(im)
            element_areas[min_g] += im.size[0] * im.size[1]

        stitched = [bg]
        for grp in element_groups:
            if grp:
                stitched.append(
                    stitch_images(
                        grp, target_area=target_area, pad_color=pad_color
                    )
                )
        return stitched

    # ── Balanced (legacy) strategy ─────────────────────────────────────
    num_groups = 2 if n <= 5 else min(max_groups, 3)

    # Greedy partitioning: assign images to the group with smallest
    # accumulated area.
    groups: list[list[Image.Image]] = [[] for _ in range(num_groups)]
    group_areas: list[int] = [0] * num_groups

    # Sort images by area descending for better balance
    indexed_all = sorted(
        enumerate(images),
        key=lambda t: t[1].size[0] * t[1].size[1],
        reverse=True,
    )
    for orig_idx, im in indexed_all:
        min_g = min(range(num_groups), key=lambda g: group_areas[g])
        groups[min_g].append(im)
        group_areas[min_g] += im.size[0] * im.size[1]

    stitched: list[Image.Image] = []
    for grp in groups:
        if grp:
            stitched.append(
                stitch_images(grp, target_area=target_area, pad_color=pad_color)
            )

    return stitched


def build_group_mapping(
    images: Sequence[Image.Image],
    max_groups: int = 3,
    background_first: bool = False,
) -> tuple[list[list[int]], int]:
    """Return ``(groups, num_groups)`` where ``groups[g]`` is the list of
    original 0-based image indices assigned to group *g*.

    This mirrors the partitioning logic of :func:`partition_and_stitch` but
    only returns the index mapping.

    Parameters
    ----------
    background_first:
        If *True*, image 0 is always placed alone in group 0 (background),
        and images 1..N-1 are distributed among the remaining groups.
    """
    n = len(images)
    if n <= max_groups:
        return [[i] for i in range(n)], n

    # ── Background-first strategy ──────────────────────────────────────
    if background_first:
        rest_max = max_groups - 1
        rest_indices = list(range(1, n))

        if len(rest_indices) <= rest_max:
            groups_out: list[list[int]] = [[0]] + [[i] for i in rest_indices]
            return groups_out, len(groups_out)

        rest_groups: list[list[int]] = [[] for _ in range(rest_max)]
        rest_areas: list[int] = [0] * rest_max

        sorted_rest = sorted(
            rest_indices,
            key=lambda i: images[i].size[0] * images[i].size[1],
            reverse=True,
        )
        for idx in sorted_rest:
            min_g = min(range(rest_max), key=lambda g: rest_areas[g])
            rest_groups[min_g].append(idx)
            rest_areas[min_g] += images[idx].size[0] * images[idx].size[1]

        for grp in rest_groups:
            grp.sort()

        all_groups = [[0]] + [g for g in rest_groups if g]
        return all_groups, len(all_groups)

    # ── Balanced (legacy) strategy ─────────────────────────────────────
    num_groups = 2 if n <= 5 else min(max_groups, 3)
    groups: list[list[int]] = [[] for _ in range(num_groups)]
    group_areas: list[int] = [0] * num_groups

    indexed = sorted(
        range(n),
        key=lambda i: images[i].size[0] * images[i].size[1],
        reverse=True,
    )

    for orig_idx in indexed:
        min_g = min(range(num_groups), key=lambda g: group_areas[g])
        groups[min_g].append(orig_idx)
        group_areas[min_g] += images[orig_idx].size[0] * images[orig_idx].size[1]

    # Sort indices within each group to preserve relative ordering
    for grp in groups:
        grp.sort()

    return [g for g in groups if g], num_groups
