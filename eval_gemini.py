import argparse
import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
from collections import defaultdict
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


DATA_TASKS_DIR = "data/tasks"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
DEFAULT_OUTPUT_DIR = "data/output/supplementary/gemini"

BASE_MODEL_PATH = os.environ.get("QWEN_BASE_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
CHECKPOINT_PATH = os.environ.get(
    "QWEN_CHECKPOINT_PATH", "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "insert-your-gemini-api-key-here")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image")
# eval.py hard-codes this evaluator. Keep it fixed so scores cannot silently
# mix models when a shell still exports an older experiment override.
GEMINI_VIESCORE_MODEL = "gemini-2.5-flash-lite"
BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://globalai.vip")
API_KEY_HEADER = os.environ.get("GEMINI_API_KEY_HEADER", "api-key")

# This identifier is written into every newly scored sidecar. Records without
# it were produced by the earlier supplementary evaluator and are rescored.
VIESCORE_EVAL_VERSION = "legacy_eval_py_v1"
VIESCORE_RESULT_FIELDS = {
    "viescore",
    "viescore_components",
    "viescore_reasoning",
    "viescore_raw",
    "viescore_model",
    "viescore_prompt",
    "viescore_protocol",
    "viescore_aggregation",
    "viescore_eval_version",
    "viescore_error",
    "viescore_rescore_error",
    "viescore_attempted_version",
    "viescore_reused_from",
}


FIXED_PROMPT = (
    "This is a visual in-context learning task. The first two images are an input "
    "and output of Task A. The third image is the input for Task B. The goal is "
    "to perform Task B on the third image and generate output image, learning "
    "from Task A."
)

QWEN_ANALYSIS_PROMPT = (
    "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\n"
    "The Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\n"
    "Please simply describe the input images, focus on the visual changes from input to output, and analyze the key differences between them.\n"
    "Don't give me long descriptions or explanations; keep it concise and to the point.\n"
    "Don't tell me exactly what the tasks are (e.g., denoising, colorization, or shadow removal); instead, use implicit words and highlight how they differ in their objectives and effects.\n"
    "Fit your answer into 3 sentences: 1) input image descriptions (what need to be done); 2) visual changes (what task A and B did); 3) differences of task A and B.\n"
    "I know you can't see output of task B, but you can guess what task it is based on the input."
)


TASK_DISPLAY_NAMES = {
    "colorization": "colorization",
    "deblurring": "deblurring",
    "dehazing": "dehazing",
    "demoireing": "demoireing",
    "denoising": "denoising",
    "deraining": "deraining",
    "harmonization": "harmonization",
    "inpainting": "inpainting",
    "light_enhancement": "low-light enhancement",
    "reflection_removal": "reflection removal",
    "shadow_removal": "shadow removal",
    "style_transfer": "style transfer",
}


TASK_DESCRIPTIONS = {
    "colorization": (
        "The target operation converts a grayscale or black-and-white image into "
        "a plausible full-color image while preserving the original structure and content."
    ),
    "deblurring": (
        "The target operation improves sharpness and restores clearer details in an image "
        "that suffers from blur."
    ),
    "dehazing": (
        "The target operation removes haze or fog-like veil effects, recovering clearer "
        "contrast, visibility, and natural colors."
    ),
    "demoireing": (
        "The target operation removes moire patterns or colored stripe artifacts while "
        "preserving the underlying image structure."
    ),
    "denoising": (
        "The target operation suppresses visible image noise while preserving real textures, "
        "edges, and scene content."
    ),
    "deraining": (
        "The target operation removes rain streaks, water drops, or rain-related occlusions "
        "while keeping the background scene consistent."
    ),
    "harmonization": (
        "The target operation adjusts a composite region so that it visually matches the "
        "surrounding background in illumination, color, and realism."
    ),
    "inpainting": (
        "The target operation fills missing or masked regions with plausible content that is "
        "consistent with the surrounding image."
    ),
    "light_enhancement": (
        "The target operation improves low-light visibility by increasing brightness and "
        "contrast while avoiding unnatural color shifts."
    ),
    "reflection_removal": (
        "The target operation removes unwanted reflection artifacts, such as glass reflections, "
        "while preserving the desired background scene."
    ),
    "shadow_removal": (
        "The target operation removes cast shadows and restores the appearance of the underlying "
        "surface or object."
    ),
    "style_transfer": (
        "The target operation changes the visual appearance, illumination, weather, season, or "
        "style of the scene while preserving its main content."
    ),
}


TASK_NAME_PATTERNS = [
    r"\bcolori[sz](?:e|ation|ing|ed)?\b",
    r"\bdeblur(?:ring|red|s)?\b",
    r"\bdehaz(?:e|ing|ed|es)?\b",
    r"\bdemoir[eé](?:ing|ed|s)?\b",
    r"\bmoire removal\b",
    r"\bdenois(?:e|ing|ed|es)?\b",
    r"\bderain(?:ing|ed|s)?\b",
    r"\binpaint(?:ing|ed|s)?\b",
    r"\bharmoni[sz](?:e|ation|ing|ed)?\b",
    r"\blow[- ]light enhancement\b",
    r"\blight enhancement\b",
    r"\breflection removal\b",
    r"\bshadow removal\b",
    r"\bstyle transfer\b",
    r"\bstyli[sz](?:e|ation|ing|ed)?\b",
]


TABLE2_TOP_TIER_PAIRS = [
    "deblurring__dehazing",
    "deblurring__deraining",
    "deblurring__demoireing",
    "demoireing__dehazing",
    "harmonization__light_enhancement",
    "inpainting__light_enhancement",
    "inpainting__style_transfer",
    "style_transfer__light_enhancement",
    "denoising__light_enhancement",
    "light_enhancement__deraining",
    "light_enhancement__shadow_removal",
    "reflection_removal__dehazing",
]

EXTRA_DIAGNOSTIC_PAIRS = [
    "inpainting__colorization",
    "colorization__style_transfer",
    "harmonization__style_transfer",
    "shadow_removal__reflection_removal",
]

RECOMMENDED_PAIRS = TABLE2_TOP_TIER_PAIRS + [
    pair for pair in EXTRA_DIAGNOSTIC_PAIRS if pair not in TABLE2_TOP_TIER_PAIRS
]


def hashed_id(*parts: object) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def task_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " "))


def mime_type(path: str) -> str:
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix == "png":
        return "image/png"
    if suffix in {"jpg", "jpeg"}:
        return "image/jpeg"
    if suffix == "webp":
        return "image/webp"
    return "image/jpeg"


def create_gemini_client():
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            headers={API_KEY_HEADER: GEMINI_API_KEY},
        ),
    )


def load_eval_data(eval_json: str) -> Dict[str, List[dict]]:
    data = read_json(eval_json)
    grouped = defaultdict(list)
    for entry in data:
        task_a = entry["taskA_input"].split("/")[0]
        task_b = entry["taskB_input"].split("/")[0]
        grouped[f"{task_a}__{task_b}"].append(entry)
    return dict(grouped)


def select_entries(
    grouped: Dict[str, List[dict]],
    pairs: Optional[List[str]],
    max_samples_per_pair: int,
    shuffle: bool,
    seed: int,
) -> Dict[str, List[dict]]:
    if pairs is None:
        pairs = [p for p in RECOMMENDED_PAIRS if p in grouped]
    rng = random.Random(seed)
    selected = {}
    for pair in pairs:
        if pair not in grouped:
            logging.warning(f"Requested pair not found in eval dataset: {pair}")
            continue
        entries = list(grouped[pair])
        if shuffle:
            rng.shuffle(entries)
        selected[pair] = entries[:max_samples_per_pair]
    return selected


def load_prompt_qwen():
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logging.info("Loading Qwen student prompt generator...")
    if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        try:
            model = model.merge_and_unload()
        except Exception as e:
            logging.warning(f"Failed to merge LoRA, using PEFT wrapper directly: {e}")
    elif os.path.isdir(CHECKPOINT_PATH) and os.path.exists(
        os.path.join(CHECKPOINT_PATH, "config.json")
    ):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CHECKPOINT_PATH, torch_dtype="auto", device_map="auto"
        )
    else:
        raise FileNotFoundError(
            "Cannot find the Qwen student checkpoint. Expected either "
            f"adapter_config.json or config.json under: {CHECKPOINT_PATH}. "
            "If your checkpoint is elsewhere, set QWEN_CHECKPOINT_PATH before running. "
            "The fixed-prompt results already generated will be reused when you rerun."
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    return model, processor


def generate_qwen_prompt(
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    model,
    processor,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, task_a_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, task_a_output),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, task_b_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {"type": "text", "text": QWEN_ANALYSIS_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(next(model.parameters()).device)

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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return FIXED_PROMPT + (output_text[0] if output_text else "")


def mask_task_terms(text: str) -> Tuple[str, List[str]]:
    found = []

    def repl(match):
        found.append(match.group(0))
        return "the visual operation"

    masked = text
    for pattern in TASK_NAME_PATTERNS:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked, sorted(set(found), key=str.lower)


def build_prompt(
    prompt_mode: str,
    task_a: str,
    task_b: str,
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    qwen_model=None,
    qwen_processor=None,
) -> Tuple[str, dict]:
    meta = {"prompt_mode": prompt_mode, "taskA": task_a, "taskB": task_b}

    if prompt_mode == "fixed":
        return FIXED_PROMPT, meta

    if prompt_mode == "task_name":
        prompt = (
            f"{FIXED_PROMPT}\n"
            f"For this diagnostic baseline, Task A is {task_name(task_a)} and Task B is {task_name(task_b)}. "
            f"Apply {task_name(task_b)} to the third image only."
        )
        return prompt, meta

    if prompt_mode == "target_desc":
        desc = TASK_DESCRIPTIONS.get(task_b, f"Apply {task_name(task_b)} to the third image.")
        prompt = (
            f"{FIXED_PROMPT}\n"
            "For this diagnostic baseline, use the following target-task description for Image 3: "
            f"{desc} Only edit the third image and preserve unrelated content."
        )
        meta["target_description"] = desc
        return prompt, meta

    if prompt_mode in {"qwen", "qwen_masked"}:
        if qwen_model is None or qwen_processor is None:
            raise ValueError(f"Prompt mode {prompt_mode} requires the Qwen student model.")
        prompt = generate_qwen_prompt(
            task_a_input, task_a_output, task_b_input, qwen_model, qwen_processor
        )
        if prompt_mode == "qwen_masked":
            prompt, masked_terms = mask_task_terms(prompt)
            meta["masked_terms"] = masked_terms
            meta["prompt_sha1"] = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        return prompt, meta

    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def build_combo_prompt(
    args,
    pair_key: str,
    combo_id: str,
    task_a: str,
    task_b: str,
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    qwen_model=None,
    qwen_processor=None,
) -> Tuple[str, dict]:
    """Build a prompt, reusing the exact saved Qwen prompt for qwen_masked."""
    if args.prompt_mode != "qwen_masked":
        return build_prompt(
            args.prompt_mode,
            task_a,
            task_b,
            task_a_input,
            task_a_output,
            task_b_input,
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
        )

    source_dir = os.path.join(args.output_dir, "qwen", pair_key, combo_id)
    source = load_prompt(source_dir)
    if source is not None:
        source_prompt, _ = source
        prompt, masked_terms = mask_task_terms(source_prompt)
        return prompt, {
            "prompt_mode": "qwen_masked",
            "taskA": task_a,
            "taskB": task_b,
            "source_prompt_mode": "qwen",
            "source_prompt_path": os.path.join(source_dir, "prompt.txt"),
            "source_prompt_sha1": hashlib.sha1(source_prompt.encode("utf-8")).hexdigest(),
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "prompt_changed": prompt != source_prompt,
            "masked_terms": masked_terms,
        }

    logging.warning(
        "%s/%s: saved Qwen prompt is missing; regenerating it before masking",
        pair_key,
        combo_id,
    )
    if args.require_saved_qwen_prompts:
        raise FileNotFoundError(
            f"Missing saved Qwen prompt: {os.path.join(source_dir, 'prompt.txt')}"
        )
    return build_prompt(
        "qwen_masked",
        task_a,
        task_b,
        task_a_input,
        task_a_output,
        task_b_input,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
    )


def _extract_image_from_parts(parts) -> Optional[Image.Image]:
    for p in parts:
        if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None):
            try:
                return Image.open(BytesIO(p.inline_data.data)).convert("RGB")
            except Exception as e:
                logging.warning(f"Could not open Gemini inline image: {e}")

        if getattr(p, "text", None):
            m = re.search(
                r"data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=\s\r\n]+)",
                p.text,
            )
            if m:
                try:
                    return Image.open(BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
                except Exception as e:
                    logging.warning(f"Could not decode Gemini base64 image: {e}")
    return None


def generate_image(
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    text_prompt: str,
) -> Optional[Image.Image]:
    from google.genai import types

    client = create_gemini_client()
    parts = [types.Part(text=text_prompt)]
    for rel_path in [task_a_input, task_a_output, task_b_input]:
        abs_path = os.path.join(DATA_TASKS_DIR, rel_path)
        with open(abs_path, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type(abs_path)))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        return _extract_image_from_parts(response.candidates[0].content.parts)
    except Exception as e:
        logging.warning(f"Gemini generation failed: {e}")
        return None


def eval_quality(gt_path: str, gen_path: str) -> Tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB").resize(gt_img.size, Image.BICUBIC)
    gt_np = np.array(gt_img)
    pred_np = np.array(gen_img)
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=255)
    return float(psnr), float(ssim)


def evaluate_viescore(
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    gen_path: str,
    task_a: str,
    task_b: str,
    tmp_dir: str,
) -> dict:
    """Run the same Gemini SC/PQ evaluator used by eval.py."""
    from google.genai import types
    from VIEScore.paper_implementation.imagen_museum.utils import (
        write_entry_to_json_file,
    )

    # Keep this text byte-for-byte aligned with eval.py::evaluate_generated.
    prompt = """
        The first two images show an example of visual task.
        The first image is the input of the first task [TASK_A_DEGRADATION], and the second is the output.
        The third image is a new input of the second task [TASK_B_DEGRADATION].
        The goal is to apply a similar visual task transfer from the first example to the new input.
        Please evaluate the fourth image, which is the model's generated output for the [TASK_B_DEGRADATION] task.
        Rate the fourth image based on two criteria:
        1. **Semantic Consistency (SC):** How well does the fourth image successfully obey the [TASK_B_DEGRADATION], similar to how the [TASK_A_DEGRADATION] was done in the example? (1-10)
        2. **Perceptual Quality (PQ):** Is the fourth image of high visual quality? (1-10)
        Return JSON strictly in this format: {{"score": [SC, PQ], "reasoning": "..."}}
    """.replace("[TASK_A_DEGRADATION]", task_a).replace("[TASK_B_DEGRADATION]", task_b)

    parts = [types.Part(text=prompt)]
    image_paths = [
        os.path.join(DATA_TASKS_DIR, task_a_input),
        os.path.join(DATA_TASKS_DIR, task_a_output),
        os.path.join(DATA_TASKS_DIR, task_b_input),
        gen_path,
    ]
    for path in image_paths:
        with open(path, "rb") as f:
            legacy_mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=legacy_mime))

    client = create_gemini_client()
    ensure_dir(tmp_dir)
    tmp_file_path = os.path.join(tmp_dir, "viescore_log.json")
    uid = hashed_id(task_a_input, task_b_input, gen_path)
    last_error = None
    for i in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_VIESCORE_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1024),
            )
            raw = response.candidates[0].content.parts[0].text
            parsed = write_entry_to_json_file(
                input_string=raw,
                uid=uid,
                prompt_input=prompt,
                vision_input=image_paths,
                output_file_name=tmp_file_path,
                give_up_parsing=False,
            )
            if parsed == "rate_limit_exceeded":
                last_error = "rate_limit_exceeded"
                break
            if parsed is not True:
                raise ValueError(f"Legacy VIEScore parser returned {parsed!r}")

            parsed_data = read_json(tmp_file_path).get(uid, {})
            scores = parsed_data.get("score", [])
            score_list = [float(x) for x in scores]
            if len(score_list) == 2:
                score = float((score_list[0] + score_list[1]) / 2)
            elif len(score_list) == 1:
                score = score_list[0]
            else:
                raise ValueError(
                    f"Legacy VIEScore parser returned {len(score_list)} scores"
                )
            return {
                "viescore": score,
                "viescore_components": score_list,
                "viescore_reasoning": parsed_data.get("reasoning", ""),
                "viescore_raw": raw,
                "viescore_model": GEMINI_VIESCORE_MODEL,
                "viescore_prompt": prompt,
                "viescore_protocol": "eval.py::evaluate_generated",
                "viescore_aggregation": "arithmetic_mean",
                "viescore_eval_version": VIESCORE_EVAL_VERSION,
            }
        except Exception as e:
            last_error = str(e)
            logging.warning(f"VIEScore attempt {i + 1} failed for {gen_path}: {e}")
    return {
        "viescore": None,
        "viescore_error": last_error,
        "viescore_model": GEMINI_VIESCORE_MODEL,
        "viescore_attempted_version": VIESCORE_EVAL_VERSION,
    }


def sidecar_path(combo_dir: str, attempt_index: int) -> str:
    return os.path.join(combo_dir, f"attempt_{attempt_index:02d}.json")


def image_path(combo_dir: str, attempt_index: int) -> str:
    return os.path.join(combo_dir, f"attempt_{attempt_index:02d}.png")


def load_attempt_record(combo_dir: str, attempt_index: int) -> Optional[dict]:
    path = sidecar_path(combo_dir, attempt_index)
    if not os.path.exists(path):
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def save_prompt(combo_dir: str, prompt: str, meta: dict) -> None:
    ensure_dir(combo_dir)
    with open(os.path.join(combo_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    write_json(os.path.join(combo_dir, "prompt_meta.json"), meta)


def load_prompt(combo_dir: str) -> Optional[Tuple[str, dict]]:
    prompt_path = os.path.join(combo_dir, "prompt.txt")
    meta_path = os.path.join(combo_dir, "prompt_meta.json")
    if not os.path.exists(prompt_path):
        return None
    with open(prompt_path, "r") as f:
        prompt = f.read()
    meta = read_json(meta_path) if os.path.exists(meta_path) else {}
    return prompt, meta


def should_evaluate_vie(mode: str, attempt_index: int, best_index: int) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if mode == "first_best":
        return attempt_index == 0 or attempt_index == best_index
    raise ValueError(f"Unknown VIEScore mode: {mode}")


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not np.isnan(float(value))


def has_current_viescore(rec: dict) -> bool:
    return (
        is_number(rec.get("viescore"))
        and rec.get("viescore_eval_version") == VIESCORE_EVAL_VERSION
        and rec.get("viescore_model") == GEMINI_VIESCORE_MODEL
    )


def clear_viescore_fields(rec: dict) -> None:
    for key in VIESCORE_RESULT_FIELDS:
        rec.pop(key, None)


def apply_viescore_result(rec: dict, result: dict) -> None:
    """Install a fresh score without destroying a usable old score on API failure."""
    if is_number(result.get("viescore")):
        clear_viescore_fields(rec)
        rec.update(result)
        return

    if is_number(rec.get("viescore")):
        rec["viescore_rescore_error"] = result.get("viescore_error")
        rec["viescore_attempted_version"] = VIESCORE_EVAL_VERSION
        return

    clear_viescore_fields(rec)
    rec.update(result)


def score_attempt_records(
    args,
    combo_dir: str,
    pair_key: str,
    combo_id: str,
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    task_a: str,
    task_b: str,
    attempt_records: List[Optional[dict]],
) -> None:
    ok_records = [
        rec
        for rec in attempt_records
        if rec
        and rec.get("status") == "ok"
        and is_number(rec.get("psnr"))
        and os.path.exists(rec.get("image") or "")
    ]
    if not ok_records:
        return

    best = max(ok_records, key=lambda rec: float(rec["psnr"]))
    best_index = int(best["attempt_index"])
    for rec in ok_records:
        attempt_index = int(rec["attempt_index"])
        if not should_evaluate_vie(
            args.evaluate_viescore, attempt_index, best_index
        ):
            continue
        if has_current_viescore(rec) and not args.force_viescore:
            continue

        logging.info(
            "%s/%s: evaluating legacy eval.py VIEScore for attempt %d",
            pair_key,
            combo_id,
            attempt_index,
        )
        result = evaluate_viescore(
            task_a_input,
            task_a_output,
            task_b_input,
            rec["image"],
            task_a,
            task_b,
            tmp_dir=combo_dir,
        )
        apply_viescore_result(rec, result)
        write_json(sidecar_path(combo_dir, attempt_index), rec)


def selected_prompts_need_qwen(args, selected: Dict[str, List[dict]]) -> bool:
    if args.viescore_only or args.prompt_mode not in {"qwen", "qwen_masked"}:
        return False

    for pair_key, entries in selected.items():
        for entry in entries:
            combo_id = hashed_id(entry["taskA_input"], entry["taskB_input"])
            own_dir = os.path.join(args.output_dir, args.prompt_mode, pair_key, combo_id)
            if load_prompt(own_dir) is not None and not args.regenerate_prompts:
                continue
            if args.prompt_mode == "qwen_masked":
                source_dir = os.path.join(args.output_dir, "qwen", pair_key, combo_id)
                if load_prompt(source_dir) is not None:
                    continue
                if args.require_saved_qwen_prompts:
                    raise FileNotFoundError(
                        f"Missing saved Qwen prompt: {os.path.join(source_dir, 'prompt.txt')}"
                    )
            return True
    return False


def process_combo(
    args,
    mode_output_dir: str,
    pair_key: str,
    entry: dict,
    qwen_model=None,
    qwen_processor=None,
) -> List[dict]:
    task_a, task_b = pair_key.split("__", 1)
    task_a_input = entry["taskA_input"]
    task_a_output = entry["taskA_output"]
    task_b_input = entry["taskB_input"]
    task_b_output = entry["taskB_output"]
    combo_id = hashed_id(task_a_input, task_b_input)

    combo_dir = os.path.join(mode_output_dir, pair_key, combo_id)
    if args.viescore_only and not os.path.isdir(combo_dir):
        logging.warning(f"{pair_key}/{combo_id}: no existing output to rescore")
        return []
    ensure_dir(combo_dir)

    prompt = None
    prompt_meta = {}
    if not args.viescore_only:
        prompt_tuple = load_prompt(combo_dir)
        if prompt_tuple is None or args.regenerate_prompts:
            prompt, prompt_meta = build_combo_prompt(
                args,
                pair_key,
                combo_id,
                task_a,
                task_b,
                task_a_input,
                task_a_output,
                task_b_input,
                qwen_model=qwen_model,
                qwen_processor=qwen_processor,
            )
            save_prompt(combo_dir, prompt, prompt_meta)
        else:
            prompt, prompt_meta = prompt_tuple

        write_json(os.path.join(combo_dir, "sample.json"), entry)

    gt_path = os.path.join(DATA_TASKS_DIR, task_b_output)
    attempt_records: List[Optional[dict]] = []
    for attempt_index in range(args.num_tries):
        rec = load_attempt_record(combo_dir, attempt_index)
        out_path = image_path(combo_dir, attempt_index)

        if args.viescore_only:
            if rec and rec.get("status") == "ok":
                recorded_image = rec.get("image") or out_path
                if os.path.exists(recorded_image):
                    rec["image"] = recorded_image
                    if not is_number(rec.get("psnr")) or not is_number(rec.get("ssim")):
                        psnr, ssim = eval_quality(gt_path, recorded_image)
                        rec["psnr"] = psnr
                        rec["ssim"] = ssim
                        write_json(sidecar_path(combo_dir, attempt_index), rec)
            attempt_records.append(rec)
            continue

        if args.prompt_mode == "qwen_masked" and not prompt_meta.get(
            "prompt_changed", True
        ):
            rec_prompt_meta = (rec or {}).get("prompt_meta") or {}
            if (
                rec
                and rec.get("status") == "ok"
                and os.path.exists(out_path)
                and rec_prompt_meta.get("prompt_sha1") == prompt_meta.get("prompt_sha1")
                and not args.force_regenerate
            ):
                attempt_records.append(rec)
                continue

            source_dir = os.path.join(args.output_dir, "qwen", pair_key, combo_id)
            source_rec = load_attempt_record(source_dir, attempt_index)
            source_image = (source_rec or {}).get("image") or image_path(
                source_dir, attempt_index
            )
            if (
                source_rec
                and source_rec.get("status") == "ok"
                and os.path.exists(source_image)
            ):
                logging.info(
                    f"{pair_key}/{combo_id}: masked prompt is unchanged; "
                    f"reusing Qwen attempt {attempt_index}"
                )
                shutil.copy2(source_image, out_path)
                rec = {
                    "combo_id": combo_id,
                    "pair_key": pair_key,
                    "attempt_index": attempt_index,
                    "status": "ok",
                    "prompt_mode": args.prompt_mode,
                    "image": out_path,
                    "psnr": source_rec.get("psnr"),
                    "ssim": source_rec.get("ssim"),
                    "taskA_input": task_a_input,
                    "taskA_output": task_a_output,
                    "taskB_input": task_b_input,
                    "taskB_output": task_b_output,
                    "prompt_meta": prompt_meta,
                    "reused_qwen_output": True,
                    "reused_qwen_sidecar": sidecar_path(source_dir, attempt_index),
                }
                for key in VIESCORE_RESULT_FIELDS:
                    if key in source_rec:
                        rec[key] = source_rec[key]
                if not is_number(rec.get("psnr")) or not is_number(rec.get("ssim")):
                    psnr, ssim = eval_quality(gt_path, out_path)
                    rec["psnr"] = psnr
                    rec["ssim"] = ssim
                if has_current_viescore(rec):
                    rec["viescore_reused_from"] = sidecar_path(
                        source_dir, attempt_index
                    )
            else:
                rec = {
                    "combo_id": combo_id,
                    "pair_key": pair_key,
                    "attempt_index": attempt_index,
                    "status": "no_image",
                    "prompt_mode": args.prompt_mode,
                    "taskA_input": task_a_input,
                    "taskA_output": task_a_output,
                    "taskB_input": task_b_input,
                    "taskB_output": task_b_output,
                    "prompt_meta": prompt_meta,
                    "error": "unchanged prompt but source Qwen attempt is unavailable",
                }
            write_json(sidecar_path(combo_dir, attempt_index), rec)
            attempt_records.append(rec)
            continue

        prompt_matches = True
        if rec and args.prompt_mode == "qwen_masked":
            recorded_prompt_meta = rec.get("prompt_meta") or {}
            prompt_matches = recorded_prompt_meta.get("prompt_sha1") == prompt_meta.get(
                "prompt_sha1"
            )

        if (
            rec is None
            or args.force_regenerate
            or not os.path.exists(out_path)
            or not prompt_matches
        ):
            if rec is not None and not prompt_matches:
                logging.info(
                    f"{pair_key}/{combo_id}: masked prompt changed; regenerating attempt {attempt_index}"
                )
            logging.info(f"{pair_key}/{combo_id}: generating attempt {attempt_index + 1}/{args.num_tries}")
            img = generate_image(task_a_input, task_a_output, task_b_input, prompt)
            if img is None:
                rec = {
                    "combo_id": combo_id,
                    "pair_key": pair_key,
                    "attempt_index": attempt_index,
                    "status": "no_image",
                    "prompt_mode": args.prompt_mode,
                    "taskA_input": task_a_input,
                    "taskA_output": task_a_output,
                    "taskB_input": task_b_input,
                    "taskB_output": task_b_output,
                }
                write_json(sidecar_path(combo_dir, attempt_index), rec)
                attempt_records.append(rec)
                continue
            img.save(out_path)
            psnr, ssim = eval_quality(gt_path, out_path)
            rec = {
                "combo_id": combo_id,
                "pair_key": pair_key,
                "attempt_index": attempt_index,
                "status": "ok",
                "prompt_mode": args.prompt_mode,
                "image": out_path,
                "psnr": psnr,
                "ssim": ssim,
                "taskA_input": task_a_input,
                "taskA_output": task_a_output,
                "taskB_input": task_b_input,
                "taskB_output": task_b_output,
                "prompt_meta": prompt_meta,
            }
            write_json(sidecar_path(combo_dir, attempt_index), rec)
        elif rec.get("status") == "ok" and (
            not is_number(rec.get("psnr")) or not is_number(rec.get("ssim"))
        ):
            psnr, ssim = eval_quality(gt_path, out_path)
            rec["psnr"] = psnr
            rec["ssim"] = ssim
            write_json(sidecar_path(combo_dir, attempt_index), rec)

        attempt_records.append(rec)

    score_attempt_records(
        args,
        combo_dir,
        pair_key,
        combo_id,
        task_a_input,
        task_a_output,
        task_b_input,
        task_a,
        task_b,
        attempt_records,
    )

    return [load_attempt_record(combo_dir, i) for i in range(args.num_tries)]


def numeric_values(records: Iterable[dict], key: str) -> List[float]:
    values = []
    for r in records:
        v = r.get(key)
        if isinstance(v, (int, float)) and not np.isnan(v):
            values.append(float(v))
    return values


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def std_or_none(values: List[float]) -> Optional[float]:
    return float(np.std(values, ddof=0)) if values else None


def median_or_none(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def summarize_combo(records: List[dict]) -> Optional[dict]:
    ok = [r for r in records if r and r.get("status") == "ok" and "psnr" in r]
    if not ok:
        return None
    ok = sorted(ok, key=lambda r: r["attempt_index"])
    first = next((rec for rec in ok if int(rec["attempt_index"]) == 0), None)
    identity = first or ok[0]
    best = max(ok, key=lambda r: r["psnr"])

    psnrs = numeric_values(ok, "psnr")
    ssims = numeric_values(ok, "ssim")
    vies = [float(rec["viescore"]) for rec in ok if has_current_viescore(rec)]
    complete_vies = len(vies) == len(ok)
    identity_prompt_meta = identity.get("prompt_meta") or {}
    base = {
        "combo_id": identity["combo_id"],
        "pair_key": identity["pair_key"],
        "num_attempts": len(ok),
        "num_viescores": len(vies),
        "first_attempt_index": 0 if first is not None else None,
        "best_attempt_index": best["attempt_index"],
        "first_psnr": first.get("psnr") if first else None,
        "first_ssim": first.get("ssim") if first else None,
        "first_viescore": (
            first.get("viescore") if first and has_current_viescore(first) else None
        ),
        "mean_psnr": mean_or_none(psnrs),
        "std_psnr": std_or_none(psnrs),
        "median_psnr": median_or_none(psnrs),
        "mean_ssim": mean_or_none(ssims),
        "std_ssim": std_or_none(ssims),
        "median_ssim": median_or_none(ssims),
        "mean_viescore": mean_or_none(vies) if complete_vies else None,
        "std_viescore": std_or_none(vies) if complete_vies else None,
        "median_viescore": median_or_none(vies) if complete_vies else None,
        "best_psnr": best.get("psnr"),
        "best_ssim": best.get("ssim"),
        "best_viescore": best.get("viescore") if has_current_viescore(best) else None,
        "best_image": best.get("image"),
        "first_image": first.get("image") if first else None,
        "taskA_input": identity.get("taskA_input"),
        "taskA_output": identity.get("taskA_output"),
        "taskB_input": identity.get("taskB_input"),
        "taskB_output": identity.get("taskB_output"),
        "prompt_changed": identity_prompt_meta.get("prompt_changed"),
        "reused_qwen_output": bool(identity.get("reused_qwen_output")),
    }
    return base


def aggregate_combo_summaries(summaries: List[dict]) -> dict:
    out = {
        "num_combos": len(summaries),
        "num_prompt_changed": sum(s.get("prompt_changed") is True for s in summaries),
        "num_prompt_unchanged": sum(s.get("prompt_changed") is False for s in summaries),
        "num_reused_qwen_output": sum(
            bool(s.get("reused_qwen_output")) for s in summaries
        ),
    }
    keys = [
        "first_psnr",
        "first_ssim",
        "first_viescore",
        "mean_psnr",
        "std_psnr",
        "median_psnr",
        "mean_ssim",
        "std_ssim",
        "median_ssim",
        "mean_viescore",
        "std_viescore",
        "median_viescore",
        "best_psnr",
        "best_ssim",
        "best_viescore",
    ]
    for key in keys:
        vals = numeric_values(summaries, key)
        out[f"avg_{key}"] = mean_or_none(vals)
        out[f"count_{key}"] = len(vals)
    return out


def load_all_attempt_records(mode_output_dir: str) -> List[dict]:
    records = []
    for root, _, files in os.walk(mode_output_dir):
        for name in files:
            if re.match(r"attempt_\d+\.json$", name):
                path = os.path.join(root, name)
                try:
                    rec = read_json(path)
                    if isinstance(rec, dict):
                        records.append(rec)
                except Exception:
                    continue
    return records


def selected_combo_keys(selected: Optional[Dict[str, List[dict]]]) -> Optional[set]:
    if selected is None:
        return None
    return {
        (
            pair_key,
            hashed_id(entry["taskA_input"], entry["taskB_input"]),
        )
        for pair_key, entries in selected.items()
        for entry in entries
    }


def write_summary(
    mode_output_dir: str,
    selected: Optional[Dict[str, List[dict]]] = None,
    num_tries: Optional[int] = None,
) -> dict:
    records = load_all_attempt_records(mode_output_dir)
    scope = selected_combo_keys(selected)
    by_combo = defaultdict(list)
    for rec in records:
        if rec.get("combo_id") and rec.get("pair_key"):
            key = (rec["pair_key"], rec["combo_id"])
            if scope is not None and key not in scope:
                continue
            if num_tries is not None:
                attempt_index = rec.get("attempt_index")
                if not isinstance(attempt_index, int) or attempt_index >= num_tries:
                    continue
            by_combo[key].append(rec)

    combo_summaries = []
    for _, recs in sorted(by_combo.items()):
        summary = summarize_combo(recs)
        if summary:
            combo_summaries.append(summary)

    by_pair = defaultdict(list)
    for s in combo_summaries:
        by_pair[s["pair_key"]].append(s)

    pair_summary = {
        pair: aggregate_combo_summaries(items) for pair, items in sorted(by_pair.items())
    }
    global_summary = aggregate_combo_summaries(combo_summaries)

    write_json(os.path.join(mode_output_dir, "combo_summary.json"), combo_summaries)
    write_json(os.path.join(mode_output_dir, "pair_summary.json"), pair_summary)
    write_json(os.path.join(mode_output_dir, "summary_global.json"), global_summary)

    log_path = os.path.join(mode_output_dir, "evaluation_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    for s in combo_summaries:
        append_jsonl(log_path, s)

    table_path = os.path.join(mode_output_dir, "summary_table.md")
    with open(table_path, "w") as f:
        f.write("| Pair | N | First PSNR | Mean PSNR | Std PSNR | Best PSNR | First VIE (coverage) | Mean VIE (coverage) | Best VIE (coverage) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for pair, stats in pair_summary.items():
            f.write(
                "| {pair} | {n} | {first_psnr} | {mean_psnr} | {std_psnr} | {best_psnr} | {first_vie} | {mean_vie} | {best_vie} |\n".format(
                    pair=pair,
                    n=stats.get("num_combos", 0),
                    first_psnr=fmt(stats.get("avg_first_psnr")),
                    mean_psnr=fmt(stats.get("avg_mean_psnr")),
                    std_psnr=fmt(stats.get("avg_std_psnr")),
                    best_psnr=fmt(stats.get("avg_best_psnr")),
                    first_vie=fmt_coverage(
                        stats.get("avg_first_viescore"),
                        stats.get("count_first_viescore"),
                        stats.get("num_combos", 0),
                    ),
                    mean_vie=fmt_coverage(
                        stats.get("avg_mean_viescore"),
                        stats.get("count_mean_viescore"),
                        stats.get("num_combos", 0),
                    ),
                    best_vie=fmt_coverage(
                        stats.get("avg_best_viescore"),
                        stats.get("count_best_viescore"),
                        stats.get("num_combos", 0),
                    ),
                )
            )
        f.write(
            "| **Global** | {n} | {first_psnr} | {mean_psnr} | {std_psnr} | {best_psnr} | {first_vie} | {mean_vie} | {best_vie} |\n".format(
                n=global_summary.get("num_combos", 0),
                first_psnr=fmt(global_summary.get("avg_first_psnr")),
                mean_psnr=fmt(global_summary.get("avg_mean_psnr")),
                std_psnr=fmt(global_summary.get("avg_std_psnr")),
                best_psnr=fmt(global_summary.get("avg_best_psnr")),
                first_vie=fmt_coverage(
                    global_summary.get("avg_first_viescore"),
                    global_summary.get("count_first_viescore"),
                    global_summary.get("num_combos", 0),
                ),
                mean_vie=fmt_coverage(
                    global_summary.get("avg_mean_viescore"),
                    global_summary.get("count_mean_viescore"),
                    global_summary.get("num_combos", 0),
                ),
                best_vie=fmt_coverage(
                    global_summary.get("avg_best_viescore"),
                    global_summary.get("count_best_viescore"),
                    global_summary.get("num_combos", 0),
                ),
            )
        )

    logging.info(f"Wrote summary to {mode_output_dir}")
    return global_summary


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def fmt_coverage(value: Optional[float], count: Optional[int], total: int) -> str:
    if value is None:
        return f"-- ({count or 0}/{total})"
    return f"{value:.3f} ({count or 0}/{total})"


def parse_pairs(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or raw.strip() == "":
        return None
    if raw.strip().lower() == "recommended":
        return list(RECOMMENDED_PAIRS)
    if raw.strip().lower() in {"table2", "top_tier"}:
        return list(TABLE2_TOP_TIER_PAIRS)
    return [x.strip() for x in raw.split(",") if x.strip()]


def run(args) -> None:
    mode_output_dir = os.path.join(args.output_dir, args.prompt_mode)
    ensure_dir(mode_output_dir)

    pairs = parse_pairs(args.pairs)
    grouped = load_eval_data(args.eval_json)
    selected = select_entries(
        grouped,
        pairs,
        args.max_samples_per_pair,
        args.shuffle,
        args.seed,
    )

    if args.summarize_only:
        write_summary(mode_output_dir, selected, args.num_tries)
        return

    qwen_model = qwen_processor = None
    if selected_prompts_need_qwen(args, selected):
        qwen_model, qwen_processor = load_prompt_qwen()

    run_meta = {
        "prompt_mode": args.prompt_mode,
        "num_tries": args.num_tries,
        "max_samples_per_pair": args.max_samples_per_pair,
        "pairs": list(selected.keys()),
        "evaluate_viescore": args.evaluate_viescore,
        "gemini_model": GEMINI_MODEL,
        "gemini_viescore_model": GEMINI_VIESCORE_MODEL,
        "viescore_eval_version": VIESCORE_EVAL_VERSION,
        "viescore_aggregation": "arithmetic_mean",
        "viescore_only": args.viescore_only,
        "base_url": BASE_URL,
        "eval_json": args.eval_json,
    }
    write_json(os.path.join(mode_output_dir, "run_meta.json"), run_meta)

    for pair_key, entries in selected.items():
        logging.info(f"Processing {pair_key}: {len(entries)} samples")
        for idx, entry in enumerate(entries, 1):
            try:
                process_combo(
                    args,
                    mode_output_dir,
                    pair_key,
                    entry,
                    qwen_model=qwen_model,
                    qwen_processor=qwen_processor,
                )
                if args.summary_every > 0 and idx % args.summary_every == 0:
                    write_summary(mode_output_dir, selected, args.num_tries)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.exception(f"Failed sample in {pair_key}: {e}")
        write_summary(mode_output_dir, selected, args.num_tries)

    write_summary(mode_output_dir, selected, args.num_tries)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Gemini prompt-condition experiments for T2T-VICL."
    )
    parser.add_argument(
        "--prompt_mode",
        required=True,
        choices=["fixed", "qwen", "qwen_masked", "task_name", "target_desc"],
        help="Prompt/baseline mode to evaluate.",
    )
    parser.add_argument("--eval_json", default=EVAL_DATASET_JSON)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pairs",
        default="recommended",
        help="Comma-separated task pairs, 'table2', 'recommended', or empty for the recommended pairs.",
    )
    parser.add_argument("--max_samples_per_pair", type=int, default=20)
    parser.add_argument("--num_tries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--summary_every",
        type=int,
        default=1,
        help="Refresh summary files after this many processed combos. Use 0 to summarize only at the end.",
    )
    parser.add_argument(
        "--evaluate_viescore",
        choices=["none", "first_best", "all"],
        default="first_best",
        help="VIEScore policy. Use 'all' only if you can afford the extra Gemini calls.",
    )
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--regenerate_prompts", action="store_true")
    parser.add_argument(
        "--require_saved_qwen_prompts",
        action="store_true",
        help="For qwen_masked, fail instead of loading Qwen when a source prompt is missing.",
    )
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--force_viescore", action="store_true")
    parser.add_argument(
        "--viescore_only",
        action="store_true",
        help="Only score existing generated images; never generate images or prompts.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
