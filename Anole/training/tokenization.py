import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

DATA_TASKS_DIR = "../../data/tasks"
input_file_path = '../../data/dataset/train_dataset.json'
output_file_path = '../../data/dataset/dataset.json'

description_template = "This is a visual in-context learning task. The first two images are an input and output of Task A: [TASK_A_DEGRADATION]. The third image is the input for Task B: [TASK_B_DEGRADATION]. The goal is to perform Task B on the third image and generate output image, learning from Task A."


def convert_data_format(input_path, output_path, template):
    """
    Loads the original data, throws away the long description,
    and replaces it with the standardized template.
    """
    print(f"Loading original data from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            actual_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    processed_data = []
    
    # Iterate over each entry in your original file
    for item in actual_data:
        # Create a new dictionary with only the fields we need
        new_item = {
            "taskA_input": item["taskA_input"],
            "taskA_output": item["taskA_output"],
            "taskB_input": item["taskB_input"],
            "taskB_output": item["taskB_output"],
            "description": template # Assign the standardized template
        }
        processed_data.append(new_item)
        
    print(f"Saving {len(processed_data)} processed items to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # Save as a JSON list, as expected by our modified tokenization.py
        json.dump(processed_data, f, indent=2) 
        
    print("Conversion complete.")
    print(f"Next step: Run tokenization.py on '{output_path}'")


def load_and_tokenize_prompts(file_path, processor, model):
    """Load and tokenize prompts from JSONL file"""
    prompts = []
    responses = []
    tokens_list = []

    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    # Iterate over the list of dictionaries
    print(f"Tokenizing {len(raw_data)} items...")
    for data in tqdm(raw_data):
        if "prompt" not in data:
            # 1. Extract task names to replace placeholders
            task_a_name = data["taskA_input"].split('/')[0]
            task_b_name = data["taskB_input"].split('/')[0]
            description = data["description"].replace("[TASK_A_DEGRADATION]", task_a_name).replace("[TASK_B_DEGRADATION]", task_b_name)

            # 2. Build prompt and response strings
            # Prompt contains 3 image placeholders
            prompt_text = f"Task A input: <image> Task A output: <image> Task B input: <image> {description}"
            # Response contains 1 image placeholder
            response_text = f"<image>"

            # 3. Build image list (3 for prompt, 1 for response)
            all_image_paths = [
                os.path.join(DATA_TASKS_DIR, data["taskA_input"]),
                os.path.join(DATA_TASKS_DIR, data["taskA_output"]),
                os.path.join(DATA_TASKS_DIR, data["taskB_input"]),
                os.path.join(DATA_TASKS_DIR, data["taskB_output"])
            ]

            images = [Image.open(img_path) for img_path in all_image_paths]
            prompt_images = images[:3]
            response_images = [images[3]]  # Keep as list

            # 4. Process prompt with its 3 images
            inputs_prompt = processor(text=prompt_text, images=prompt_images, padding=False,
                                      return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
            inputs_prompt_ids = inputs_prompt['input_ids']

            if prompt_images:
                pixel_values_prompt = inputs_prompt["pixel_values"]
                image_tokens_prompt = model.get_image_tokens(pixel_values_prompt)
                special_image_mask_prompt = inputs_prompt_ids == 8711  # Image token ID
                image_tokens_prompt = image_tokens_prompt.to(inputs_prompt_ids.device, inputs_prompt_ids.dtype)
                inputs_prompt_ids = inputs_prompt_ids.masked_scatter(special_image_mask_prompt, image_tokens_prompt)

            # 5. Process response with its 1 image
            inputs_response = processor(text=response_text, images=response_images, padding=False,
                                        return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
            inputs_response_ids = inputs_response['input_ids']

            if response_images:
                pixel_values_response = inputs_response["pixel_values"]
                image_tokens_response = model.get_image_tokens(pixel_values_response)
                special_image_mask_response = inputs_response_ids == 8711  # Image token ID
                image_tokens_response = image_tokens_response.to(inputs_response_ids.device, inputs_response_ids.dtype)
                inputs_response_ids = inputs_response_ids.masked_scatter(special_image_mask_response, image_tokens_response)

            # 6. Combine prompt and response tokens
            input_ids = inputs_prompt_ids[0].tolist() + [8710] + inputs_response_ids[0].tolist()[1:] + [2]

            prompts.append(prompt_text)
            responses.append(response_text)
            tokens_list.append(input_ids)

        elif "prompt" in data:
            if 'images' in data:
                images = [Image.open(img_path) for img_path in data['images']]
                inputs_prompt = processor(data['prompt'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                inputs_response = processor(data['response'], images=images, padding=False,
                                            return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)

                inputs_prompt_ids = inputs_prompt['input_ids']
                inputs_response_ids = inputs_response['input_ids']

                if data['images'] is not None:
                    pixel_values = inputs_response["pixel_values"]
                    image_tokens = model.get_image_tokens(pixel_values)
                    special_image_mask = inputs_response_ids == 8711  # Image token ID
                    image_tokens = image_tokens.to(inputs_response_ids.device, inputs_response_ids.dtype)
                    inputs_response_ids = inputs_response_ids.masked_scatter(special_image_mask, image_tokens)

                input_ids = inputs_prompt_ids[0].tolist() + [8710] + inputs_response_ids[0].tolist()[1:] + [2]

            else:
                inputs_prompt = processor(data['prompt'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)
                inputs_response = processor(data['response'], padding=False, return_tensors="pt", return_for_text_completion=True).to("cuda", dtype=torch.bfloat16)

                inputs_prompt_ids = inputs_prompt['input_ids']
                inputs_response_ids = inputs_response['input_ids']

                input_ids = inputs_prompt_ids[0].tolist() + [8710] + inputs_response_ids[0].tolist()[1:] + [2]

            prompts.append(data['prompt'])
            responses.append(data['response'])
            tokens_list.append(input_ids)

    return prompts, responses, tokens_list


def save_tokenized_to_jsonl(prompts, responses, tokens_list, output_path):
    """Save tokenized prompts to JSONL file"""
    with open(output_path, 'w') as f:
        for prompt, response, tokens in zip(prompts, responses, tokens_list):
            output = {
                "prompt": prompt,
                "response": response,
                "tokens": tokens,
            }
            json.dump(output, f)
            f.write('\n')


# Example usage:
def tokenize_jsonl_file(input_file, output_file, model_path):
    """
    Main function to tokenize a JSONL file of prompts

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file with tokenized prompts
        model_path: Path to the model (required for processor)
    """
    # Load model and processor
    processor = ChameleonProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load and tokenize prompts
    prompts, responses, tokens_list = load_and_tokenize_prompts(
        input_file, processor, model.model
    )

    # Save tokenized results
    save_tokenized_to_jsonl(
        prompts, responses, tokens_list, output_file
    )

    print(f"Tokenized {len(prompts)} prompts and saved to {output_file}")


convert_data_format(input_file_path, output_file_path,
                    description_template)
tokenize_jsonl_file(output_file_path, "../../data/dataset/tokenized_dataset.jsonl",
                    model_path="../../.cache/anole-7b")
