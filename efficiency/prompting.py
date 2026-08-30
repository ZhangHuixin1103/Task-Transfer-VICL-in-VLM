from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


INSTRUCT_TEXT = (
    "This is a visual in-context learning task. The first two images are an input "
    "and output of Task A. The third image is the input for Task B. The goal is to "
    "perform Task B on the third image and generate output image, learning from Task A."
)

RELATION_REQUEST = (
    "You are an expert in analyzing image processing tasks. Below are two vision tasks, "
    "A and B.\nThe Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the "
    "third image Picture 3 is input of Task B.\nPlease simply describe the input images, "
    "focus on the visual changes from input to output, and analyze the key differences "
    "between them.\nDon't give me long descriptions or explanations; keep it concise and "
    "to the point.\nDon't tell me exactly what the tasks are (e.g., denoising, colorization, "
    "or shadow removal); instead, use implicit words and highlight how they differ in their "
    "objectives and effects.\nFit your answer into 3 sentences: 1) input image descriptions "
    "(what need to be done); 2) visual changes (what task A and B did); 3) differences of "
    "task A and B.\nI know you can't see output of task B, but you can guess what task it is "
    "based on the input."
)


def generate_text_prompt(
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    model,
    processor,
    data_tasks_dir: str | Path,
    input_resolution: int | None = None,
) -> str:
    """Run the trained Qwen prompt protocol, optionally at a controlled image size."""
    from qwen_vl_utils import process_vision_info

    data_tasks_dir = Path(data_tasks_dir)
    paths = [task_a_input, task_a_output, task_b_input]
    opened_images: list[Image.Image] = []
    if input_resolution is None:
        image_sources: list[str | Image.Image] = [
            str(data_tasks_dir / path) for path in paths
        ]
    else:
        image_sources = []
        for path in paths:
            with Image.open(data_tasks_dir / path) as source:
                image = source.convert("RGB").resize(
                    (input_resolution, input_resolution),
                    Image.Resampling.BICUBIC,
                )
            opened_images.append(image)
            image_sources.append(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_sources[0],
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": image_sources[1],
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": image_sources[2],
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {"type": "text", "text": RELATION_REQUEST},
            ],
        }
    ]
    try:
        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    finally:
        for image in opened_images:
            image.close()
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            do_sample=True,
            use_cache=True,
        )
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return INSTRUCT_TEXT + output_text[0] if output_text else INSTRUCT_TEXT
