"""RedBench evaluation script using official Gemini API (Google Gen AI SDK)."""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


def get_gemini_client() -> genai.Client:
    """Create Gemini client using API key from environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please export your API key first, e.g.:\n"
            "export GEMINI_API_KEY='your_new_api_key'"
        )
    return genai.Client(api_key=api_key)


client = get_gemini_client()

MODEL_NAME = "gemini-3-flash-preview"


def extract_scores_and_average(entry: str) -> float | None:
    """Extract scores from entry string and compute average."""
    if not isinstance(entry, str):
        return None

    lines = entry.splitlines()
    scores = []

    for line in lines:
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].strip().isdigit():
            scores.append(int(parts[1].strip()))

    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_averages(result_json_dict: dict) -> dict:
    """Compute averages for all entries in result dict."""
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
    return result


def compute_edit_type_averages(score_dict: dict, meta_list: list) -> dict:
    """Compute averages grouped by edit type."""
    edit_type_scores = defaultdict(list)

    for idx, score in score_dict.items():
        meta = meta_list[int(idx)]
        edit_type = meta.get("task")
        if edit_type is not None:
            edit_type_scores[edit_type].append(score)

    averaged_by_type = {
        etype: round(sum(scores) / len(scores), 2)
        for etype, scores in edit_type_scores.items()
        if scores
    }
    return averaged_by_type


def load_prompts(prompts_json_path: str) -> dict:
    """Load prompts from JSON file."""
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_for_gemini(image_path: str):
    """Load image using PIL."""
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def pil_to_part(img: Image.Image, mime_type: str = "image/png") -> types.Part:
    """Convert PIL image to Gemini SDK inline image Part."""
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(
        data=buf.getvalue(),
        mime_type=mime_type,
    )


def call_gemini(
    original_image_path: str,
    result_image_path: str,
    edit_prompt: str,
    edit_type: str,
    prompts: dict,
) -> str:
    """Call Gemini API to evaluate image edit."""
    try:
        # Load images
        original_image = load_image_for_gemini(original_image_path)
        result_image = load_image_for_gemini(result_image_path)

        if original_image is None or result_image is None:
            return {"error": "Image conversion failed"}

        prompt_template = prompts[edit_type]
        full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

        # Build content with images and text
        contents = [
            pil_to_part(original_image),
            "这是原图A",
            pil_to_part(result_image),
            "这是编辑后的图B。请进行评估。",
            full_prompt,
        ]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=8192,
            ),
        )

        if hasattr(response, "text") and response.text:
            return response.text

        try:
            return response.candidates[0].content.parts[0].text
        except Exception:
            return "Error: Empty response text"

    except Exception as e:
        print(f"Error in calling Gemini API: {e}")
        return f"Error: {str(e)}"


def process_single_item(
    idx: int,
    item: dict,
    result_img_folder: str,
    prompts: dict,
) -> tuple[str, str]:
    """Process a single edit item."""
    # Build result image path
    task = item["task"]
    item_id = item["id"]

    result_img_name = f"{task}-{item_id}.png"
    result_img_path = os.path.join(result_img_folder, task, result_img_name)

    # Original image path
    origin_img_path = item["source"]

    # Select prompt: prefer Chinese, otherwise English
    if args.lang == "cn":
        edit_prompt = item.get("a_to_b_instructions", "")
    elif args.lang == "eng":
        edit_prompt = item.get("a_to_b_instructions_eng", "")

    # Edit type
    res = call_gemini(origin_img_path, result_img_path, edit_prompt, task, prompts)
    return str(idx), res


def process_json(
    edit_infos: list,
    result_img_folder: str,
    num_threads: int,
    prompts: dict,
) -> dict:
    """Process all edit items with thread pool."""
    results = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(
                process_single_item, idx, item, result_img_folder, prompts
            ): idx
            for idx, item in enumerate(edit_infos)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing edits",
        ):
            idx = future_to_idx[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing idx {idx}: {e}")
                results[str(idx)] = {"error": str(e)}

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_img_folder",
        type=str,
        required=True,
        help="Folder containing generated result images (Root)",
    )
    parser.add_argument(
        "--edit_jsonl",
        type=str,
        required=True,
        help="Path to edit info jsonl file",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        required=True,
        help="Path to prompts json file",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=20,
        help="Number of concurrent threads",
    )
    arg_parser.add_argument(
        "--lang", 
        type=str, 
        choices=["cn", "eng"], 
        default="cn", 
        help="Language choice: cn or eng (default: cn)"
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)

    # Read jsonl file
    edit_infos = []
    with open(args.edit_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                edit_infos.append(json.loads(line))

    print(f"Loaded {len(edit_infos)} entries")

    data = process_json(edit_infos, args.result_img_folder, args.num_threads, prompts)
    averaged_data = compute_averages(data)
    averaged_result = compute_edit_type_averages(averaged_data, edit_infos)

    if averaged_result:
        scores = list(averaged_result.values())
        final_score = sum(scores) / len(scores)
    else:
        final_score = 0

    print("\n--- Results by Category ---")
    print(json.dumps(averaged_result, indent=4, ensure_ascii=False))
    print(f"\nFinal Overall Score: {final_score:.4f}")

    results_path = os.path.join(args.result_img_folder, "evaluation_details.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    scores_path = os.path.join(args.result_img_folder, "score_summary.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_score": final_score,
                "averaged_result": averaged_result,
                "averaged_data": averaged_data,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()