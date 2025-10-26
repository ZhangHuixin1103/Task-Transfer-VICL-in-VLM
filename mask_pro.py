import argparse
import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from io import BytesIO
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from google import genai
from google.genai import types
from peft import PeftModel
from PIL import Image, ImageFilter
from qwen_vl_utils import process_vision_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)
from transformers.generation.stopping_criteria import (StoppingCriteria,
                                                       StoppingCriteriaList)

from GrAInS.src.attribution.gradient.vlm_grad import \
    get_token_attributions_contrastive
from GrAInS.src.utils.config import MODEL_NAME_MAP
from GrAInS.src.utils.model import load_vlm_model_and_processor
from VIEScore.paper_implementation.imagen_museum.utils import \
    write_entry_to_json_file

# Add VIEScore path
viescore_path = '/data1/tzz/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Set constants
DATA_TASKS_DIR = "data/tasks"
TRAIN_DATASET_JSON = "data/dataset/train_dataset.json"
EVAL_DATASET_JSON = "data/dataset/eval_dataset.json"
OUTPUT_DIR = "data/output"
TMP_DIR = "data/tmp"
VIS_DIR = "data/vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL = "qwen-2.5-vl-3b-instruct"
CHECKPOINT_PATH = "Qwen2.5-VL/qwen-vl-finetune/output/checkpoint-latest"

GEMINI_API_KEY = "sk-2LE9SvYG170QGDDX1ajIUlsuVxt1bqY9nY92BZAKvSZlPWFL"
GEMINI_MODEL = "gemini-2.0-flash-preview-image-generation"
BASE_URL = "https://globalai.vip"
API_KEY_HEADER = "api-key"

TRUNCATE_LEN = 2000

def _shorten(text: str, n: int = TRUNCATE_LEN) -> str:
    if not text:
        return ""
    try:
        if len(text) <= n:
            return text
        return text[:n] + f"... (truncated, total {len(text)} chars)"
    except Exception:
        return text

def hashed_id(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]

def create_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            base_url=BASE_URL,
            headers={API_KEY_HEADER: GEMINI_API_KEY}
        )
    )

def generate_eval_dataset():
    """Generate new image combinations per task pair not in train data."""
    if os.path.exists(EVAL_DATASET_JSON):
        logging.info(f"Loading existing {EVAL_DATASET_JSON}")
        with open(EVAL_DATASET_JSON, 'r') as f:
            eval_data = json.load(f)
    else:
        random.seed(42)
        with open(TRAIN_DATASET_JSON, 'r') as f:
            train_data = json.load(f)

        existing_combos = set()
        task_pairs = set()
        for entry in train_data:
            combo = (entry['taskA_input'],
                     entry['taskA_output'],
                     entry['taskB_input'])
            existing_combos.add(combo)
            taskA = entry['taskA_input'].split('/')[0]
            taskB = entry['taskB_input'].split('/')[0]
            task_pairs.add((taskA, taskB))

        all_tasks = [d for d in os.listdir(DATA_TASKS_DIR)
                     if os.path.isdir(os.path.join(DATA_TASKS_DIR, d))]
        task_pairs_dict = {}
        for task in all_tasks:
            input_dir = os.path.join(DATA_TASKS_DIR, task, 'input')
            output_dir = os.path.join(DATA_TASKS_DIR, task, 'output')
            inputs = [f for f in os.listdir(input_dir)
                      if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(input_dir) else []
            outputs = [f for f in os.listdir(output_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(output_dir) else []

            pairs = []
            for inp in inputs:
                stem, ext = os.path.splitext(inp)
                matching_out = next(
                    (out for out in outputs if os.path.splitext(out)[0] == stem), None)
                if matching_out:
                    pairs.append({
                        "input": os.path.join(task, 'input', inp),
                        "output": os.path.join(task, 'output', matching_out)
                    })
            task_pairs_dict[task] = pairs

        eval_data = []
        for taskA, taskB in task_pairs:
            a_pairs = task_pairs_dict.get(taskA, [])
            b_pairs = task_pairs_dict.get(taskB, [])

            candidates = []
            for a_pair in a_pairs:
                for b_pair in b_pairs:
                    combo = (a_pair['input'],
                             a_pair['output'],
                             b_pair['input'])
                    if combo not in existing_combos:
                        candidates.append({
                            "taskA_input": a_pair['input'],
                            "taskA_output": a_pair['output'],
                            "taskB_input": b_pair['input'],
                            "taskB_output": b_pair['output']
                        })

            random.shuffle(candidates)
            selected = candidates[:100]
            if len(selected) < 100:
                logging.warning(f"Only {len(selected)} combos for {taskA} -> {taskB}!")
            eval_data.extend(selected)

        with open(EVAL_DATASET_JSON, 'w') as f:
            json.dump(eval_data, f, indent=4)

        logging.info(f"Generated {len(eval_data)} new combos for eval dataset.")

    grouped = {}
    for entry in eval_data:
        taskA = entry['taskA_input'].split('/')[0]
        taskB = entry['taskB_input'].split('/')[0]
        pair_key = f"{taskA}__{taskB}"
        grouped.setdefault(pair_key, []).append(entry)

    return eval_data, grouped

def generate_text_prompt(taskA_input, taskA_output, taskB_input, model, processor, use_qwen=True, fixed_prompt=None):
    if not use_qwen and fixed_prompt is None:
        return fixed_prompt or "You are an expert in analyzing image processing tasks. Fix the third image and generate an output image."

    elif not use_qwen and fixed_prompt is not None:
        taskA = taskA_input.split('/')[0]
        taskB = taskB_input.split('/')[0]

        prompt = fixed_prompt.replace('[TASK_A_DEGRADATION]', taskA).replace('[TASK_B_DEGRADATION]', taskB)

        logging.info(f"Using fixed prompt:\n{prompt}")
        return prompt

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskA_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskA_output),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "image",
                    "image": os.path.join(DATA_TASKS_DIR, taskB_input),
                    "min_pixels": 12544,
                    "max_pixels": 401408,
                },
                {
                    "type": "text",
                    "text": "You are an expert in analyzing image processing tasks. Below are two vision tasks, A and B.\nThe Picture 1 and 2 belong to Task A, 1 is input and 2 is output; the third image Picture 3 is input of Task B.\nPlease analyze and describe the key differences between the two tasks shortly. Focus on the degradation to be removed, for example: 'Remove rain streaks', 'Enhance low light areas', or 'Remove shadows'.\nI know you can't see output of task B, but you can guess what task it is based on shortcoming of input.",
                },
            ],
        },
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
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print(f"Generated prompt:\n{output_text[0] if output_text else ''}")
    return output_text[0] if output_text else ""

def _fallback_simple_mask(b_input_path: str, save_dir: str, blur_r: int = 51, th: float = 0.30, smooth_r: int = 9) -> str:
    img = Image.open(b_input_path).convert("RGB")
    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    blur = np.array(img.filter(ImageFilter.GaussianBlur(
        radius=blur_r)).convert("L")).astype(np.float32) / 255.0

    eps = 1e-6
    sal = (blur - gray) / (blur + eps)
    sal = np.clip(sal, 0.0, 1.0)

    sal_img = Image.fromarray(
        (sal * 255).astype(np.uint8), "L").resize(img.size, Image.Resampling.LANCZOS)
    binary = (np.array(sal_img) > (th * 255)).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, "L").filter(
        ImageFilter.GaussianBlur(radius=smooth_r))

    stem = os.path.splitext(os.path.basename(b_input_path))[0]
    out_path = os.path.join(save_dir, f"mask_{stem}.png")
    mask_img.save(out_path)
    return out_path

def _build_contrastive_texts(text_prompt: str, taskA: str, taskB: str) -> Tuple[str, str, str]:
    attribution_prompt = (
        f"You are given two related vision tasks.\n"
        f"- Task A ({taskA}): Image A_in → Image A_out.\n"
        f"- Task B ({taskB}): Image B_in is provided below.\n"
        f"Learn from Task A and decide where Image B_in should be restored/fixed."
    )
    pos_response = (
        f"Successfully perform {taskB} on Image B_in, following Task A -> B transfer."
        f"The goal is: {text_prompt}"
    )
    neg_response = (
        "Fail to restore Image B_in or introduce artifacts, color shifts, texture distortions, or hallucinations."
    )
    return attribution_prompt, pos_response, neg_response

def _select_image_token_scores(pos_scores, pos_ids, tokenizer) -> np.ndarray:
    try:
        ids = pos_ids[0].detach().cpu().numpy().tolist()
    except Exception:
        return np.array(pos_scores, dtype=float)

    image_like_tokens = {
        "<image>", "<img>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<image_patch>"
    }

    keep_idx = []
    for i, tid in enumerate(ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        tok_norm = (tok or "").strip().lower()
        if any(tag in tok_norm for tag in image_like_tokens):
            keep_idx.append(i)

    if not keep_idx:
        return np.array(pos_scores, dtype=float)

    return np.array([pos_scores[i] for i in keep_idx], dtype=float)

def _normalize_scores(scores):
    smin, smax = np.min(scores), np.max(scores)
    if smax - smin > 0:
        return (scores - smin) / (smax - smin)
    return np.zeros_like(scores)

def _scores_to_mask(scores: np.ndarray, target_size: Tuple[int, int], percentile: int) -> Optional[Image.Image]:
    n = scores.shape[0]
    if n < 16:
        return None

    g = int(np.sqrt(n))
    if g * g != n:
        g = int(np.floor(np.sqrt(n)))
        n2 = g * g
        if n2 < 16:
            return None
        scores = scores[:n2]

    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smax > smin:
        norm = (scores - smin) / (smax - smin)
    else:
        norm = np.zeros_like(scores)

    grid = (norm.reshape(g, g) * 255).astype(np.uint8)
    heat = Image.fromarray(grid, "L").resize(target_size, Image.Resampling.LANCZOS)

    arr = np.array(heat).astype(np.float32) / 255.0
    thr = np.percentile(arr, percentile)
    binary = (arr >= thr).astype(np.uint8) * 255
    return Image.fromarray(binary, "L")

def generate_mask(
    taskA: str,
    taskB: str,
    text_prompt: str,
    taskA_input: Optional[str] = None,
    taskA_output: Optional[str] = None,
    taskB_input: str = None,
    qwen_model=None,
    qwen_processor=None,
    method: str = "integrated_gradients",
    threshold_percentile: int = 60,
) -> Optional[str]:
    b_abs = os.path.join(DATA_TASKS_DIR, taskB_input)
    if not os.path.exists(b_abs):
        logging.error(f"[Mask] Task B input not found: {b_abs}")
        return None

    can_grains = (
        (qwen_model is not None)
        and (qwen_processor is not None)
        and (taskA_input is not None)
        and (taskA_output is not None)
    )

    if can_grains:
        try:
            a_in_abs = os.path.join(DATA_TASKS_DIR, taskA_input)
            a_out_abs = os.path.join(DATA_TASKS_DIR, taskA_output)
            imgA_in = Image.open(a_in_abs).convert("RGB")
            imgA_out = Image.open(a_out_abs).convert("RGB")
            imgB_in = Image.open(b_abs).convert("RGB")
        except Exception as e:
            logging.warning(f"[Mask][GrAInS] Failed to open images ({e}).")
            return _fallback_simple_mask(b_abs, TMP_DIR)

        attribution_prompt, pos_resp, neg_resp = _build_contrastive_texts(
            text_prompt, taskA, taskB)

        try:
            attrib = get_token_attributions_contrastive(
                model=qwen_model,
                processor=qwen_processor,
                image=imgB_in,
                prompt=attribution_prompt,
                pos_response=pos_resp,
                neg_response=neg_resp,
                method=method,
            )
            pos_scores, pos_ids = attrib["pos"]

            scores = _select_image_token_scores(pos_scores, pos_ids,
                                                qwen_processor.tokenizer)

            mask_img = _scores_to_mask(
                scores, imgB_in.size, threshold_percentile)
            if mask_img is None:
                logging.warning("[Mask][GrAInS] Cannot form a valid patch grid.")
                return _fallback_simple_mask(b_abs, TMP_DIR)

            out_path = os.path.join(
                TMP_DIR, f"mask_grains_{os.path.basename(taskB_input)}.png")
            mask_img.save(out_path)
            logging.info(f"[Mask] Saved GrAInS-based mask → {out_path}")
            return out_path

        except Exception as e:
            logging.warning(f"[Mask][GrAInS] Attribution failed ({e}).")
            return _fallback_simple_mask(b_abs, TMP_DIR)

    return _fallback_simple_mask(b_abs, TMP_DIR)

def _extract_image_from_parts(parts):
    for p in parts:
        if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None):
            try:
                return Image.open(BytesIO(p.inline_data.data))
            except Exception as e:
                logging.warning(f"Could not open inline_data as image: {e}")
                pass

        if getattr(p, "text", None):
            m = re.search(
                r"data:image/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=\s\r\n]+)",
                p.text
            )
            if m:
                b64_str = m.group(1)
                try:
                    raw_bytes = base64.b64decode(b64_str)
                    return Image.open(BytesIO(raw_bytes))
                except Exception as e:
                    logging.warning(f"Could not decode or open base64 image: {e}")
                    pass

    return None

def generate_image(taskA_input, taskA_output, taskB_input, text_prompt, mask_path=None, gt_ext='.jpg'):
    client = create_gemini_client()

    mime_a_in = "image/png" if taskA_input.endswith('.png') else "image/jpeg"
    mime_a_out = "image/png" if taskA_output.endswith('.png') else "image/jpeg"
    mime_b_in = "image/png" if taskB_input.endswith('.png') else "image/jpeg"

    image1_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskA_input), 'rb').read(),
        mime_type=mime_a_in)
    image2_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskA_output), 'rb').read(),
        mime_type=mime_a_out)
    image3_part = types.Part.from_bytes(
        data=open(os.path.join(DATA_TASKS_DIR, taskB_input), 'rb').read(),
        mime_type=mime_b_in)

    prompt_text = text_prompt
    if mask_path and os.path.exists(mask_path):
        prompt_text += "\nThe final image, the mask, indicates the regions on Image 3 to be restored. White areas can be restored to remove defects, while black areas are likely be kept unchanged.\nRestore Image 3 by fixing the defects mainly in the white regions of the mask. Leave the black regions of the mask untouched, and do not change colors, textures, or objects in these areas."

    contents = [types.Part(text=prompt_text),
                image1_part, image2_part, image3_part]

    if mask_path and os.path.exists(mask_path):
        mask_part = types.Part.from_bytes(data=open(mask_path, 'rb').read(),
                                          mime_type="image/png")
        contents.append(mask_part)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        parts = response.candidates[0].content.parts

        for p in parts:
            if getattr(p, "text", None):
                logging.info(f"Gemini returned text:\n---\n{_shorten(p.text)}\n---")
                break

        image = _extract_image_from_parts(parts)
        if image:
            return image
        else:
            logging.warning("Gemini response had no image (neither inline_data nor base64 in text).")
            return None

    except Exception as e:
        logging.warning(f"Gemini generation failed: {e}")

    return None

def eval_quality(gt_path, gen_path):
    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    gen_img = gen_img.resize(gt_img.size, Image.BICUBIC)
    gt_np = np.array(gt_img)
    pred_np = np.array(gen_img)

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=255)

    return psnr, ssim

def evaluate_generated(gt_path, gen_path, taskA_input, taskA_output, taskB_input, taskA, taskB):
    psnr, ssim = eval_quality(gt_path, gen_path)

    viescore_prompt = f"""
        The first two images show an example of visual task.
        The first image is the input of the first task [TASK_A_DEGRADATION], and the second is the output.
        The third image is a new input of the second task [TASK_B_DEGRADATION].
        The goal is to apply a similar visual task transfer from the first example to the new input.
        Please evaluate the fourth image, which is the model's generated output for the [TASK_B_DEGRADATION] task.
        Rate the fourth image based on two criteria:
        1. **Semantic Consistency (SC):** How well does the fourth image successfully obey the [TASK_B_DEGRADATION], similar to how the [TASK_A_DEGRADATION] was done in the example? (1-10)
        2. **Perceptual Quality (PQ):** Is the fourth image of high visual quality? (1-10)
        Return JSON strictly in this format: {{"score": [SC, PQ], "reasoning": "..."}}
    """.replace('[TASK_A_DEGRADATION]', taskA).replace('[TASK_B_DEGRADATION]', taskB)

    image_list = [
        os.path.join(DATA_TASKS_DIR, taskA_input),
        os.path.join(DATA_TASKS_DIR, taskA_output),
        os.path.join(DATA_TASKS_DIR, taskB_input),
        gen_path
    ]
    parts = [types.Part(text=viescore_prompt)]
    for path in image_list:
        mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))

    client = create_gemini_client()

    viescore = 0.0
    is_verified = False
    tries, max_tries = 0, 2
    tmp_file_path = os.path.join(TMP_DIR, "viescore_log.json")
    uid = hashed_id(taskA_input, taskB_input, gen_path)

    while not is_verified and tries < max_tries:
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024
                )
            )
            result_text = resp.candidates[0].content.parts[0].text
            print("Raw result from Gemini:\n", result_text)

            is_verified = write_entry_to_json_file(
                input_string=result_text,
                uid=uid,
                prompt_input=viescore_prompt,
                vision_input=image_list,
                output_file_name=tmp_file_path,
                give_up_parsing=False
            )

            if is_verified is True:
                with open(tmp_file_path, "r") as f:
                    data = json.load(f)
                scores = data[uid].get("score", [])
                if len(scores) == 2:
                    sc, pq = scores
                    viescore = (sc + pq) / 2
                elif len(scores) == 1:
                    viescore = scores[0]
                break
            elif is_verified == "rate_limit_exceeded":
                logging.warning("Gemini rate limit exceeded.")
                break
            else:
                logging.warning(f"Parsing failed on try {tries+1}")

        except Exception as e:
            logging.warning(f"Error during Gemini evaluation: {e}")

        tries += 1

    if not is_verified:
        logging.error(f"Failed to get valid VIEScore for {gen_path}")

    return psnr, ssim, viescore

def compute_signed_attributions(grads, embeddings):
    return torch.einsum("ij,ij->i", grads, embeddings).detach().cpu().numpy()

def apply_integrated_gradients_contrastive(
    model,
    pos_embeds,
    neg_embeds,
    attention_mask_pos,
    attention_mask_neg,
    target_token_id_pos,
    target_token_id_neg,
    steps=20
):
    model.eval()
    T = pos_embeds.shape[1]

    joint_input = torch.cat([pos_embeds, neg_embeds], dim=1)
    baseline = torch.zeros_like(joint_input)

    def contrastive_fn(joint_embeds):
        pos_embed = joint_embeds[:, :T, :]
        neg_embed = joint_embeds[:, T:, :]

        pos_logits = model(inputs_embeds=pos_embed, attention_mask=attention_mask_pos).logits[:, -1, :]
        neg_logits = model(inputs_embeds=neg_embed, attention_mask=attention_mask_neg).logits[:, -1, :]

        log_p_pos = F.log_softmax(pos_logits, dim=-1)[:, target_token_id_pos]
        log_p_neg = F.log_softmax(neg_logits, dim=-1)[:, target_token_id_neg]

        return log_p_pos - log_p_neg

    ig = IntegratedGradients(contrastive_fn)
    attributions = ig.attribute(joint_input, baselines=baseline, n_steps=steps)

    return attributions[:, :T, :][0], attributions[:, T:, :][0]

def get_token_attributions_contrastive_multi(model, processor, images, prompt, pos_response, neg_response, method="integrated_gradients"):
    model.eval()
    model.zero_grad()

    pos_full = prompt + " " + pos_response
    neg_full = prompt + " " + neg_response

    pos_input = processor(images=images, text=pos_full, return_tensors="pt").to(model.device)
    neg_input = processor(images=images, text=neg_full, return_tensors="pt").to(model.device)

    pos_ids, neg_ids = pos_input["input_ids"], neg_input["input_ids"]
    pos_mask, neg_mask = pos_input["attention_mask"], neg_input["attention_mask"]

    embed_fn = model.get_input_embeddings()
    pos_embed = embed_fn(pos_ids).detach().requires_grad_(True)
    neg_embed = embed_fn(neg_ids).detach().requires_grad_(True)

    pos_target = pos_ids[0, -1].item()
    neg_target = neg_ids[0, -1].item()

    if method == "integrated_gradients":
        pos_grad, neg_grad = apply_integrated_gradients_contrastive(
            model, pos_embed, neg_embed, pos_mask, neg_mask, pos_target, neg_target
        )
        pos_scores = pos_grad.sum(dim=-1).detach().cpu().numpy()
        neg_scores = neg_grad.sum(dim=-1).detach().cpu().numpy()

    else:
        raise ValueError(f"Unknown attribution method: {method}")

    return {
        "pos": (pos_scores, pos_ids),
        "neg": (neg_scores, neg_ids)
    }

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id, device):
        self.stop_token_id = stop_token_id
        self.device = device

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[0, -1] == self.stop_token_id).item()

class InterleavedGenerator:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.attn_implementation = "flash_attention_2"

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)

        self.boi_token_id = self.config.boi_token_id
        self.eoi_token_id = self.config.eoi_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = 1

        self.image_conditioned_allowed = set([i for i in range(4, 8196)]) | {
            self.config.bos_token_id,
            self.boi_token_id,
            self.eoi_token_id,
        }

        self.model.setup_cfg(
            guidance_scale_full=2.0,
            guidance_scale_image=1.2,
            guidance_scale_negative=0.0,
            guidance_scale_original_prompt=5.0,
            config=self.config,
            cfg_config="no"
        )

        self.original_prompt_tokens = None

    def _prepare_cfg_batch(self, token_ids, cfg_type="normal"):
        negative_prompt = "text in the image, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        negative_tokens = self.processor.tokenizer.encode(negative_prompt, add_special_tokens=False)

        batch_token_ids = []

        if cfg_type == "normal":
            batch_token_ids.append(token_ids)
            batch_token_ids.append([self.boi_token_id])
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)

        max_len = max(len(seq) for seq in batch_token_ids)
        attention_masks = []

        for i, seq in enumerate(batch_token_ids):
            padding_length = max_len - len(seq)
            if padding_length > 0:
                batch_token_ids[i] = [self.pad_token_id] * padding_length + seq
                attention_masks.append([0] * padding_length + [1] * len(seq))
            else:
                attention_masks.append([1] * len(seq))

        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        return input_ids, attention_mask

    def generate_interleaved(
        self,
        prompt_tokens: list,
        original_prompt_tokens: list = None,
        max_length: int = 5000,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_images: int = 4,
        cfg_type: str = "normal",
        mode: str = "general"
    ):
        self.model.cfg_config = "no"

        all_tokens = prompt_tokens.copy()

        num_images_generated = 0
        generation_segments = []

        while len(all_tokens) < max_length and num_images_generated < max_images:
            current_input_ids = torch.tensor([all_tokens], device=self.device)

            stop_at_boi = StopAtSpecificTokenCriteria(self.boi_token_id, self.device)
            stop_at_eos = StopAtSpecificTokenCriteria(self.eos_token_id, self.device)

            text_output = self.model.generate(
                input_ids=current_input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList([stop_at_boi, stop_at_eos]),
                multimodal_generation_mode="interleaved-text-image",
                pad_token_id=self.pad_token_id
            )

            new_tokens = text_output[0][len(all_tokens):].tolist()
            all_tokens.extend(new_tokens)

            if all_tokens[-1] == self.eos_token_id:
                break

            if all_tokens[-1] == self.boi_token_id:
                self.model.cfg_config = cfg_type

                cfg_input_ids, cfg_attention_mask = self._prepare_cfg_batch(all_tokens, cfg_type=cfg_type)

                image_output = self.model.generate(
                    input_ids=cfg_input_ids,
                    attention_mask=cfg_attention_mask,
                    max_new_tokens=1026,
                    temperature=temperature,
                    do_sample=True,
                    multimodal_generation_mode="image-only",
                    pad_token_id=self.pad_token_id
                )

                new_image_tokens = image_output[0][len(cfg_input_ids[0]):].tolist()[:1025]
                all_tokens.extend(new_image_tokens)

                num_images_generated += 1

                self.model.cfg_config = "no"

        return {
            "tokens": all_tokens,
            "num_images": num_images_generated,
            "total_length": len(all_tokens)
        }

def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"

    tokens = tokens[0]
    segments = []
    current_segment = []
    in_image_seg = False
    for token in tokens:
        if token == boi:
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=tokens.device).reshape(1, -1)))
            current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=tokens.device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=tokens.device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=tokens.device).reshape(1, -1)))
    return segments

def decode_image_from_tokens(processor, image_tokens):
    # Anole uses model.decode_image, but since we use Thinking, assume processor or model has decode method
    # Adjust based on actual repo; for placeholder, assume it's processor.decode_image if available
    if hasattr(processor, 'decode_image'):
        image = processor.decode_image(image_tokens)
    else:
        # Fallback or error
        raise NotImplementedError("Decode method not found")
    return image

def _create_heat_map_overlay(input_image_path, scores, percentile, combo_id, type='pos'):
    img = Image.open(input_image_path).convert("RGB")
    target_size = img.size
    mask_img = _scores_to_mask(scores, target_size, percentile)
    if mask_img is None:
        logging.warning("Could not create mask for visualization.")
        return

    heat = mask_img.convert("RGB")
    heat_arr = np.array(heat)
    heat_arr[:,:,1:] = 0  # Red channel only

    heat_img = Image.fromarray(heat_arr).convert("RGBA")
    heat_img.putalpha(128)  # Semi-transparent

    overlay = Image.alpha_composite(img.convert("RGBA"), heat_img)

    vis_path = os.path.join(VIS_DIR, f"{combo_id}_{type}_overlay.png")
    overlay.save(vis_path)
    logging.info(f"Saved {type} contribution visualization to {vis_path}")

def run_evaluation(args):
    eval_data, grouped = generate_eval_dataset()
    final_results = {}

    if args.use_qwen_for_prompt:
        if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_config.json")):
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_PATH, torch_dtype="auto", device_map="auto"
            )
            prompt_qwen_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            try:
                prompt_qwen_model = prompt_qwen_model.merge_and_unload()
            except Exception as e:
                logging.warning(f"Failed to merge LoRA: {e}")
        else:
            prompt_qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                CHECKPOINT_PATH, torch_dtype="auto", device_map="auto"
            )

        prompt_qwen_model.eval()
        prompt_qwen_processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    else:
        prompt_qwen_model = None
        prompt_qwen_processor = None

    generator = InterleavedGenerator("GAIR/twgi-subgoal-anole-7b")

    for pair_key, entries in grouped.items():
        logging.info(f"Processing pair: {pair_key}")
        pair_res_dir = os.path.join(OUTPUT_DIR, pair_key)
        os.makedirs(pair_res_dir, exist_ok=True)
        log_path = os.path.join(pair_res_dir, "evaluation_log.jsonl")

        existing_combo_ids = set()
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    existing_combo_ids.add(entry['combo_id'])

        with open(log_path, 'a') as log_file:
            pair_best_scores = []

            for entry in entries:
                taskA, taskB = pair_key.split('__', 1)
                taskA_input = entry['taskA_input']
                taskA_output = entry['taskA_output']
                taskB_input = entry['taskB_input']
                taskB_output = entry['taskB_output']
                taskB_gt_path = os.path.join(DATA_TASKS_DIR, taskB_output)
                gt_ext = os.path.splitext(taskB_gt_path)[1]

                combo_id = hashed_id(taskA_input, taskB_input)

                a_name = os.path.basename(taskA_input)
                b_name = os.path.basename(taskB_input)
                final_path = os.path.join(pair_res_dir, f"{a_name}_{b_name}_{combo_id}{gt_ext}")

                if os.path.exists(final_path):
                    if combo_id in existing_combo_ids:
                        logging.info(f"COMPLETE: Skipping combo {combo_id}, image and metrics already exist.")
                        continue
                    else:
                        logging.info(f"RESUMING: Found image for {combo_id}, calculating and logging metrics...")
                        try:
                            psnr, ssim, viescore = evaluate_generated(taskB_gt_path, final_path,
                                                                      taskA_input, taskA_output,
                                                                      taskB_input, taskA, taskB)
                            log_entry = {
                                "combo_id": combo_id,
                                "final_image": final_path,
                                "psnr": psnr,
                                "ssim": ssim,
                                "viescore": viescore
                            }
                            log_file.write(json.dumps(log_entry) + '\n')
                            pair_best_scores.append(log_entry)
                            log_file.flush()
                            os.fsync(log_file.fileno())
                            logging.info(f"SUCCESS: Logged metrics for existing image {combo_id}.")
                        except Exception as e:
                            logging.error(f"FAILURE: Could not evaluate existing image {final_path}. Error: {e}")
                        continue

                logging.info(f"STARTING: Processing new combo {combo_id}.")
                combo_tmp_dir = os.path.join(TMP_DIR, pair_key, combo_id)
                os.makedirs(combo_tmp_dir, exist_ok=True)

                text_prompt = generate_text_prompt(taskA_input, taskA_output,
                                                   taskB_input,
                                                   model=prompt_qwen_model,
                                                   processor=prompt_qwen_processor,
                                                   use_qwen=args.use_qwen_for_prompt,
                                                   fixed_prompt=args.fixed_prompt)

                imgA_in = Image.open(os.path.join(DATA_TASKS_DIR, taskA_input)).convert("RGB")
                imgA_out = Image.open(os.path.join(DATA_TASKS_DIR, taskA_output)).convert("RGB")
                imgB_in = Image.open(os.path.join(DATA_TASKS_DIR, taskB_input)).convert("RGB")

                images = [imgA_in, imgA_out, imgB_in]

                attribution_prompt, pos_resp, neg_resp = _build_contrastive_texts(text_prompt, taskA, taskB)

                attrib = get_token_attributions_contrastive_multi(
                    model=generator.model,
                    processor=generator.processor,
                    images=images,
                    prompt=attribution_prompt,
                    pos_response=pos_resp,
                    neg_response=neg_resp,
                    method="integrated_gradients"
                )

                pos_scores, pos_ids = attrib["pos"]
                neg_scores, neg_ids = attrib["neg"]

                pos_third_scores = _select_image_token_scores(pos_scores, pos_ids, generator.processor.tokenizer)
                neg_third_scores = _select_image_token_scores(neg_scores, neg_ids, generator.processor.tokenizer)

                _create_heat_map_overlay(os.path.join(DATA_TASKS_DIR, taskB_input), pos_third_scores, 60, combo_id, 'pos')
                _create_heat_map_overlay(os.path.join(DATA_TASKS_DIR, taskB_input), neg_third_scores, 60, combo_id, 'neg')

                norm_scores = _normalize_scores(pos_third_scores)

                generation_prompt = f"Task A input: <image> Task A output: <image> Task B input: <image> {text_prompt} Generate Task B output:"

                inputs = generator.processor(generation_prompt, images=images, padding=False, return_tensors="pt", return_for_text_completion=True).to(generator.device, dtype=torch.bfloat16)

                input_ids = inputs['input_ids']
                if 'pixel_values' in inputs:
                    pixel_values = inputs["pixel_values"]
                    image_tokens = generator.model.get_image_tokens(pixel_values)
                    special_image_mask = input_ids == 8711 
                    image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
                    input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

                prompt_tokens = input_ids[0].tolist() + [generator.boi_token_id]

                original_prompt = f"Generate the output image for Task B based on the example."
                original_tokens = generator.processor.tokenizer.encode(original_prompt, add_special_tokens=False)

                input_ids_tensor = torch.tensor([prompt_tokens], device=generator.device)
                input_embeds = generator.model.get_input_embeddings()(input_ids_tensor)

                mask = special_image_mask[0].cpu().numpy()
                positions = np.where(mask)[0]
                blocks = []
                current = []
                for p in positions:
                    if not current or p == current[-1] + 1:
                        current.append(p)
                    else:
                        blocks.append(current)
                        current = [p]
                blocks.append(current)

                third_indices = blocks[2] if len(blocks) == 3 else []

                alpha = 0.5
                for i, idx in enumerate(third_indices):
                    if norm_scores[i] > 0:
                        input_embeds[0, idx] *= (1 + alpha * norm_scores[i])

                best_psnr = -np.inf
                best_gen_path = None
                for i in range(args.num_tries):
                    result = generator.generate_interleaved(
                        prompt_tokens=prompt_tokens,
                        original_prompt_tokens=original_tokens,
                        max_length=6144,
                        temperature=1.0,
                        max_images=1,
                        cfg_type="normal",
                        mode="general"
                    )

                    tokens_tensor = torch.tensor([result["tokens"]], device=generator.device)
                    segments = split_token_sequence(tokens_tensor, generator.boi_token_id, generator.eoi_token_id)

                    generated_image = None
                    for seg_type, seg_tokens in segments[::-1]:
                        if seg_type == "image_seg":
                            generated_image = decode_image_from_tokens(generator.processor, seg_tokens)
                            break

                    if generated_image:
                        temp_path = os.path.join(combo_tmp_dir, f"gen_{i}{gt_ext}")
                        generated_image.save(temp_path)
                        curr_psnr, _ = eval_quality(taskB_gt_path, temp_path)
                        logging.info(f"Attempt {i+1}: Saved to {temp_path}, PSNR: {curr_psnr:.2f}")
                        if curr_psnr > best_psnr:
                            best_psnr = curr_psnr
                            best_gen_path = temp_path
                            logging.info(f"Attempt {i+1}: New best image found!")

                if best_gen_path:
                    shutil.move(best_gen_path, final_path)

                    psnr, ssim, viescore = evaluate_generated(taskB_gt_path, final_path, taskA_input, taskA_output, taskB_input, taskA, taskB)
                    log_entry = {
                        "combo_id": combo_id,
                        "final_image": final_path,
                        "psnr": psnr,
                        "ssim": ssim,
                        "viescore": viescore
                    }
                    log_file.write(json.dumps(log_entry) + '\n')
                    pair_best_scores.append(log_entry)
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    logging.info(f"Combo {combo_id}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, VIEScore={viescore:.2f}")

                    shutil.rmtree(combo_tmp_dir, ignore_errors=True)

            if pair_best_scores:
                avg_psnr = np.mean([s['psnr'] for s in pair_best_scores])
                avg_ssim = np.mean([s['ssim'] for s in pair_best_scores])
                avg_viescore = np.mean([s['viescore'] for s in pair_best_scores])
                pair_metrics_path = os.path.join(pair_res_dir, "evaluation_results.json")
                with open(pair_metrics_path, 'w') as f:
                    json.dump({
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "avg_viescore": avg_viescore
                    }, f, indent=4)
                final_results[pair_key] = {
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_viescore": avg_viescore
                }

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info("Evaluation completed. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICL Evaluation Pipeline")
    parser.add_argument("--use_qwen_for_prompt", action="store_true",
                        default=False, help="Use Qwen for generating text prompt")
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Fixed text prompt if not using Qwen")
    parser.add_argument("--use_mask", action="store_true",
                        default=False, help="Use mask for generation")
    parser.add_argument("--num_tries", type=int, default=5,
                        help="Number of generation attempts per combination")
    args = parser.parse_args()

    run_evaluation(args)
