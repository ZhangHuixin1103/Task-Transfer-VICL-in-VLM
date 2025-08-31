import json
import os
import random
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)
from transformers.utils.quantization_config import BitsAndBytesConfig

# 1. Configuration Section

# Data and output path configuration
DATA_ROOT = Path("./data/tasks")
OUTPUT_DIR = Path("./data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task configuration
TRAIN_RATIO = 0.3

# Local VLM Model Configuration
LOCAL_MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# IMPORTANT: Set a small number for testing before running the full dataset
# Set this to a small number (e.g., 10) to verify the code works.
NUM_SAMPLES_TO_PROCESS = 100000

# Load the SentenceTransformer model globally
print("Loading the semantic similarity model...")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# 2. Load Local VLM Model and Tokenizer Globally
print(
    f"Loading local VLM: {LOCAL_MODEL_ID}. This will take a significant amount of time and VRAM...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_ID, trust_remote_code=True)

# Configuration for 4-bit Quantization to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_ID, trust_remote_code=True)
print("Local VLM loaded successfully to the device.")


# 3. Function Definitions

def sample_task_data(task_name: str, task_path: Path, train_ratio: float, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Samples data for a single task, creates a training list file, and returns the sampled pairs.
    """
    print(f"Sampling data for task '{task_name}'...")
    input_dir = task_path / "input"
    output_dir_task = task_path / "output"

    if not input_dir.is_dir() or not output_dir_task.is_dir():
        print(
            f"Warning: Input or output directory for task '{task_name}' not found. Skipping.")
        return []

    files = sorted([p.name for p in input_dir.glob('*') if p.is_file()])
    pairs = [(input_dir / f, output_dir_task / f)
             for f in files if (output_dir_task / f).exists()]

    if not pairs:
        print(
            f"Warning: No valid input/output image pairs found in '{task_name}'.")
        return []

    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]

    output_txt_path = output_dir / f"train_list_{task_name}.txt"
    with open(output_txt_path, "w") as f:
        for inp_path, out_path in train_pairs:
            f.write(f"{inp_path.name}\n")

    print(f"Sampled {len(train_pairs)} pairs out of {len(pairs)}.")
    print(f"Training list saved to: {output_txt_path}")
    return train_pairs


def query_local_vlm(image_paths: list[Path], prompt: str) -> str:
    """
    Queries the locally hosted Qwen2-VL model with a list of images and a text prompt.
    """
    messages = [{"role": "user", "content": []}]

    for path in image_paths:
        messages[0]["content"].append(
            {"type": "image", "image": f"file:///{str(path.absolute())}"})

    messages[0]["content"].append({"type": "text", "text": prompt})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        print(f"Error during local model generation: {e}")
        return "Error: Failed to generate response from local model."


def load_processed_data(json_path: Path) -> list:
    """
    Loads data from a potentially incomplete JSON file.
    An incomplete file is one that starts with '[' but doesn't end with ']'.
    """
    if not json_path.exists() or os.path.getsize(json_path) == 0:
        return []

    print(f"Found existing data file at {json_path}. Attempting to resume.")
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Handle cases: empty file, file with only '[', or a complete JSON
    if not content or content == '[':
        return []

    # If the file is incomplete (missing closing ']'), we fix it for parsing
    if content.startswith('[') and not content.endswith(']'):
        # Find the last valid '}' to avoid parsing errors from a partially written object
        last_brace_pos = content.rfind('}')
        if last_brace_pos == -1:
            return [] # No complete objects found
        content = content[:last_brace_pos + 1] + ']' # Truncate and close the array
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: Could not parse existing JSON file. Starting from scratch.")
        return []


def main():
    """Main execution function to generate the comparative dataset."""
    all_tasks = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    if len(all_tasks) < 2:
        print("Error: Could not find at least two task directories.")
        return

    taskA_path, taskB_path = random.sample(all_tasks, 2)
    taskA_name, taskB_name = taskA_path.name, taskB_path.name
    trainA_pairs = sample_task_data(taskA_name, taskA_path, TRAIN_RATIO, OUTPUT_DIR)
    trainB_pairs = sample_task_data(taskB_name, taskB_path, TRAIN_RATIO, OUTPUT_DIR)

    if not trainA_pairs or not trainB_pairs:
        print("Sampling list is empty. Terminating.")
        return

    combos = [(*pairA, *pairB) for pairA in trainA_pairs for pairB in trainB_pairs]
    combos_to_process = combos[:NUM_SAMPLES_TO_PROCESS]
    
    output_json_path = OUTPUT_DIR / f"train_dataset_{taskA_name}_{taskB_name}.json"

    # RESUME
    processed_data = load_processed_data(output_json_path)
    processed_combos_set = set()
    for item in processed_data:
        # Create a unique key for each processed combination to check for existence
        key = (item['taskA_input'], item['taskA_output'], item['taskB_input'], item['taskB_output'])
        processed_combos_set.add(key)
    
    is_resuming = len(processed_combos_set) > 0
    if is_resuming:
        print(f"Resuming. Found {len(processed_combos_set)} previously generated entries.")

    print(f"Total combinations to process: {len(combos_to_process)}. Starting generation...")

    # Open the file in 'w' mode to overwrite the (potentially broken) file with a clean slate,
    # starting with the already processed data. This is safer than appending.
    with open(output_json_path, "w", encoding='utf-8') as f:
        # Write the opening bracket
        f.write("[\n")
        
        # First, write back all the data we successfully loaded
        for i, item in enumerate(processed_data):
            json.dump(item, f, indent=2, ensure_ascii=False)
            if i < len(processed_data) - 1:
                f.write(",\n")
        
        # Now, iterate through the combinations and process only the new ones
        is_first_new_write = True
        for (a_in, a_out, b_in, b_out) in tqdm(combos_to_process, desc="Generating Descriptions"):
            current_key = (
                str(a_in.relative_to(DATA_ROOT)),
                str(a_out.relative_to(DATA_ROOT)),
                str(b_in.relative_to(DATA_ROOT)),
                str(b_out.relative_to(DATA_ROOT))
            )
            
            # Skip if this combination is already processed
            if current_key in processed_combos_set:
                continue

            # This is a new item, so we process it
            prompt = (
                "You are an expert in analyzing image processing tasks. Below are two tasks, A and B, each with an input and an output image. "
                "The first two images belong to Task A, and the next two images belong to Task B. "
                "Please analyze and describe the key differences between them. Focus on the target goal, the type of degradation in the input, and the visual changes from input to output."
            )
            description = query_local_vlm([a_in, a_out, b_in, b_out], prompt)

            result = {
                "taskA_input": current_key[0],
                "taskA_output": current_key[1],
                "taskB_input": current_key[2],
                "taskB_output": current_key[3],
                "description": description
            }

            # Add a comma if we are adding to existing content, or if it's not the first new item
            if is_resuming or not is_first_new_write:
                f.write(",\n")

            json.dump(result, f, indent=2, ensure_ascii=False)
            is_first_new_write = False

        # Close the JSON array after the loop completes
        f.write("\n]")

    print(f"Data generation complete! The dataset is saved to: {output_json_path}")


if __name__ == "__main__":
    main()
