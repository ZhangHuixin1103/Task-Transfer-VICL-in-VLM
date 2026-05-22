# 🚩 RedBench (REDEdit-Bench)


<p align="center">
  <a href="https://huggingface.co/datasets/FireRedTeam/REDEdit-Bench" target="_blank"><img alt="Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RedBench-ffc107?color=ffc107&logoColor=white" style="display: inline-block;"/></a>
  <a href="https://github.com/FireRedTeam/FireRed-Image-Edit"><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
  <a href='https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode'><img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg" alt="License"></a>
  <a href="https://arxiv.org/abs/2602.13344" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv"></a>
</p> 

## 🔥 Introduction

**RedBench** (also known as REDEdit-Bench) is a comprehensive benchmark designed to evaluate the capabilities of current image editing models. 

Our main goal is to build more diverse scenarios and editing instructions that better align with human language. We collected over 3,000 images from the internet, and after careful expert-designed selection, we constructed **1,673 bilingual (Chinese–English) editing pairs** across **15 categories**.

📢 **Note on Dataset Size**: The original benchmark described in the paper consists of 1,673 image pairs. However, due to strict redistribution licensing restrictions on certain commercial assets, the public release version has been curated to **1,542 pairs**. This ensures full compliance with copyright laws while maintaining the diversity and quality of the benchmark.

## ✨ Key Features

- **🗣️ Human-Aligned Instructions**: Diverse scenarios and editing instructions that closely mimic real-world human usage.
- **🌐 Bilingual Support**: Full support for both Chinese and English editing instructions.
- **🛡️ Quality Assurance**: Carefully curated by experts from a massive collection of source images.
- **🧩 Diverse Tasks**: Covers 15 distinct categories including Object Addition, Removal, Replacement, Style Transfer, and more.

## 📂 Data Structure & Examples

The dataset is organized in JSONL format. Each entry contains the image source, bilingual instructions, and the specific task category.

### Task Categories

The benchmark covers 15 different task categories:

| Category | Count | Description |
|----------|-------|-------------|
| add | 143 | Object Addition |
| adjust | 156 | Attribute Adjustment |
| background | 91 | Background Modification |
| beauty | 79 | Beauty Enhancement |
| color | 99 | Color Modification |
| compose | 100 | Image Composition |
| extract | 95 | Element Extraction |
| lowlevel | 47 | Low-level Processing |
| motion | 78 | Motion Addition |
| portrait | 102 | Portrait Editing |
| remove | 147 | Object Removal |
| replace | 140 | Object Replacement |
| stylize | 92 | Style Transfer |
| text | 123 | Text Editing |
| viewpoint | 50 | Viewpoint Change |
| all | 1542 | All Tasks |

### Sample Data

```json
{"id": "1", "source": "redbench/add/add-1.png", "a_to_b_instructions": "在图片中绿色植物上增加一只七星瓢虫", "a_to_b_instructions_eng": "Add a seven-spotted ladybug on the green plant in the picture", "task": "add"}
{"id": "2", "source": "redbench/add/add-2.png", "a_to_b_instructions": "在咖啡杯里加一个白色心形拉花", "a_to_b_instructions_eng": "Add a white heart-shaped latte art in the coffee cup", "task": "add"}
{"id": "3", "source": "redbench/add/add-3.png", "a_to_b_instructions": "在马路上增加一个穿运动服跑步的男人", "a_to_b_instructions_eng": "Add a man running in sportswear on the road", "task": "add"}
```

# Generate Images

Before evaluating the model, you first need to use the provided JSONL file (which contains metadata information) along with the original image files to generate the corresponding edited images by editing model.

We provide the inference script `redbench_infer.py` for generating edited images. This script supports multi-GPU distributed inference using Accelerate.

## Dependencies
Install required dependencies:
```bash
pip install accelerate diffusers transformers pillow tqdm
```
Then download our dataset [REDEdit_Bench.tar](https://huggingface.co/datasets/FireRedTeam/REDEdit-Bench/resolve/main/REDEdit_Bench.tar?download=true). Please download the tar file and extract it.

## Usage

To run the inference script, use the following command:

```bash
accelerate launch --num_processes <num_gpus> redbench_infer.py --model-path <path_to_model> --jsonl-path <path_to_redbench_jsonl> --save-path <path_to_save_results>
```

### Arguments:
- `--model-path`: Path to the model. Default is `FireRedTeam/FireRed-Image-Edit-1.0`.
- `--lora-name`: Path to LoRA weights (optional).
- `--save-path`: Directory to save the generated images (required).
- `--jsonl-path`: Path to the JSONL file containing edit instructions (required).
- `--edit-task`: Specific task to process (e.g., `add`, `remove`, `stylize`). Default is `all`.
- `--save-key`: Key name for saving result path. Default is `result`.
- `--seed`: Random seed. Default is 43.
- `--lang`: Instruction language, cn or eng (default: cn).

### Example:

```bash
# Generate all edited images using 8 GPUs
accelerate launch --num_processes 8 redbench_infer.py \
    --model-path FireRedTeam/FireRed-Image-Edit-1.1 \
    --jsonl-path ./redbench.jsonl \
    --save-path ./edited_images_cn \
    --edit-task all \
    --lang cn
```

## Example Input/Output

### Input

A JSONL file containing image edit instructions (`redbench.jsonl`):

```jsonl
{"id": "1", "source": "redbench/add/add-1.png", "a_to_b_instructions": "在图片中绿色植物上增加一只七星瓢虫", "a_to_b_instructions_eng": "Add a seven-spotted ladybug on the green plant in the picture", "task": "add"}
{"id": "2", "source": "redbench/add/add-2.png", "a_to_b_instructions": "在咖啡杯里加一个白色心形拉花", "a_to_b_instructions_eng": "Add a white heart-shaped latte art in the coffee cup", "task": "add"}
{"id": "3", "source": "redbench/adjust/adjust-144.png", "a_to_b_instructions": "将天空的颜色调成更深的蓝色", "a_to_b_instructions_eng": "Change the sky color to a deeper blue", "task": "adjust"}
```

A folder containing original images:

```folder
├── redbench                    
│   ├── add     
│   │   ├── add-1.png                 
│   │   ├── add-2.png                 
│   │   ├── ...                 
│   ├── adjust                             
│   │   ├── adjust-144.png
│   │   ├── ...
│   ├── ...
```

### Output

A folder containing edited images:

```folder
# Without --multi-folder option:
├── edited_images                    
│   ├── 1.png                 
│   ├── 2.png            
│   ├── 3.png           
│   ...            
│   ├── result.jsonl

# With --multi-folder option:
├── edited_images                    
│   ├── add
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── ...
│   ├── adjust
│   │   ├── 144.png
│   │   ├── ...
│   ...
│   ├── result.jsonl
``` 

# Image Editing Evaluation using Gemini-3-Flash

This project evaluates image editing processes using the **Gemini-3-Flash API**. The system processes a set of original and edited images, comparing them according to a predefined set of criteria, such as instruction adherence, image-editing quality, and detail preservation.

We provide the evaluation script `redbench_eval.py` for automated evaluation using Gemini.

## Overview

The goal of this project is to evaluate the quality of image editing processes using Gemini. The evaluation criteria include:
- **Instruction Adherence**: The edit must match the specified editing instructions.
- **Image-editing Quality**: The edit should appear seamless and natural.
- **Detail Preservation**: Regions not specified for editing should remain unchanged.

## Evaluation Criteria by Task Category

Different task categories use different evaluation metrics:

| Task Category | Metrics |
|---------------|---------|
| add, remove, replace, compose, extract | Prompt Compliance, Visual Seamlessness, Physical & Detail Fidelity |
| adjust, color, lowlevel | Prompt Compliance, Visual Seamlessness, Physical & Detail Fidelity |
| background, viewpoint | Prompt Compliance, Visual Seamlessness, Physical & Detail Fidelity |
| beauty, portrait | Prompt Compliance, Visual Seamlessness, Physical & Detail Fidelity |
| stylize | Style Fidelity, Content Preservation, Rendering Quality |
| motion | Prompt Compliance, Motion Realism, Visual Seamlessness |
| text | Text Fidelity, Visual Consistency, Background Preservation |

## Dependencies

```bash
pip install google-generativeai pillow tqdm
```

## Setup

1. **Gemini API Key**: Set your Gemini API key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

2. **Images and JSON File**: You will need:
   - A folder containing the edited images (`--result_img_folder`).
   - A JSONL file containing edit instructions and metadata (`--edit_json`).
   - A JSON file containing evaluation prompts for each task category (`--prompts_json`).

## Usage

To run the evaluation script, use the following command:

```bash
python redbench_eval.py --result_img_folder <path_to_edited_images> --edit_json <path_to_redbench_jsonl> --prompts_json <path_to_prompts_json> --lang <language>
```

### Arguments:
- `--result_img_folder`: The directory containing the edited images (required).
- `--edit_json`: Path to the JSONL file containing edit instructions and metadata (required).
- `--prompts_json`: Path to the JSON file containing evaluation prompts for each task category (required).
- `--num_threads`: Number of concurrent threads. Default is 50.
- `--lang`: Instruction language, cn or eng (default: cn).

### Example:

```bash
python redbench_eval.py \
    --result_img_folder ./edited_images \
    --edit_json ./redbench.jsonl \
    --prompts_json ./prompts.json \
    --num_threads 50 \
    --lang cn
```

## Example Input/Output

### Input

A JSONL file containing image edit instructions (`redbench.jsonl`):

```jsonl
{"id": "1", "source": "redbench/add/add-1.png", "a_to_b_instructions": "在图片中绿色植物上增加一只七星瓢虫", "a_to_b_instructions_eng": "Add a seven-spotted ladybug on the green plant in the picture", "task": "add"}
{"id": "2", "source": "redbench/add/add-2.png", "a_to_b_instructions": "在咖啡杯里加一个白色心形拉花", "a_to_b_instructions_eng": "Add a white heart-shaped latte art in the coffee cup", "task": "add"}
{"id": "3", "source": "redbench/adjust/adjust-144.png", "a_to_b_instructions": "将天空的颜色调成更深的蓝色", "a_to_b_instructions_eng": "Change the sky color to a deeper blue", "task": "adjust"}
```

A JSON file containing evaluation prompts for each task category (`prompts_json`):

```json
{
  "add": "\nYou are a data rater specializing in grading object addition edits. You will be given two images ...",
  "remove": "\nYou are a data rater specializing in grading object removal edits. You will be given two images ...",
  "adjust": "\nYou are a data rater specializing in grading attribute alteration edits. You will be given two images ....",
  "stylize": "\nYou are a data rater specializing in grading style transfer edits. You will be given an input image, a reference style...",
  ...
}
```

A folder containing edited images (with `--multi-folder` option from inference):

```folder
├── edited_images                    
│   ├── add
│   │   ├── 1.png                 
│   │   ├── 2.png
│   │   ├── ...                
│   ├── adjust                             
│   │   ├── 144.png
│   │   ...
│   ...                 
```

### Output

The script automatically computes and saves results in the result folder:

1. `result.json` - Detailed evaluation for each image:
```json
{
    "0": "Brief reasoning: A seven-spotted ladybug was successfully added on the green plant with natural color and placement.\nPrompt Compliance: 5\nVisual Seamlessness: 4\nPhysical & Detail Fidelity: 5",
    "1": "Brief reasoning: A white heart-shaped latte art was added in the coffee cup with good blending.\nPrompt Compliance: 5\nVisual Seamlessness: 4\nPhysical & Detail Fidelity: 4",
    "2": "Brief reasoning: The sky color was changed to a deeper blue with smooth transition.\nPrompt Compliance: 5\nVisual Seamlessness: 4\nPhysical & Detail Fidelity: 5",
    ...
}
```

2. `score.json` - Final scores including per-category averages and overall score:
```json
{
    "final_score": 4.3,
    "averaged_result": {
        "add": 4.5,
        "adjust": 4.2,
        "background": 3.8,
        ...
    },
    "averaged_data": {
        "0": 4.67,
        "1": 4.33,
        "2": 4.67,
        ...
    }
}
```

The `redbench_eval.py` script automatically computes:
1. Individual image scores (extracted from Gemini responses)
2. Per-category averages (averaged_result)
3. Overall final score (average of all category scores)

See the Output section above for the complete score.json structure.

## 🧩 License
**REDEdit-Bench** is released under the [Creative Commons Attribution–NonCommercial–NoDerivatives (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

- ✅ **Free for academic research purposes only**
- ❌ **Commercial use is prohibited**

🖼️ **Data Source:** All images included in REDEdit-Bench were legally purchased and obtained through official channels to ensure copyright compliance.

*By using this dataset, you agree to comply with the applicable license terms.*
