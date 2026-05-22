"""RedBench inference script for image editing."""

import json
import os
import os.path as osp

import torch
from accelerate import Accelerator
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from tqdm import tqdm


def main(args):
    """Main inference function."""
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(
            f"[rank {accelerator.process_index}] will distribute the inference "
            f"process to {accelerator.num_processes} processes"
        )
        os.makedirs(args.save_path, exist_ok=False)

    accelerator.wait_for_everyone()

    data_list = []
    with open(args.jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            data = json.loads(stripped_line)
            if args.edit_task == data["task"] or args.edit_task == "all":
                sub_folder = os.path.join(args.save_path, data["task"])
                if accelerator.is_main_process:
                    os.makedirs(sub_folder, exist_ok=True)
                
                src_name = osp.basename(data["source"]) 
                img_save_path = os.path.join(sub_folder, src_name)

                data[args.save_key] = img_save_path
                data_list.append(data)

    data_list_process = [
        x
        for i, x in enumerate(data_list)
        if i % accelerator.num_processes == accelerator.process_index
    ]

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    if args.lora_name is not None:
        pipe.load_lora_weights(args.lora_name, adapter_name="demo")

    accelerator.wait_for_everyone()

    for data_line in tqdm(
        data_list_process,
        desc="Generating",
        disable=not accelerator.is_main_process,
    ):
        if args.lang == 'cn':
            instruction = data_line.get("a_to_b_instructions", "")
        elif args.lang == 'eng':
            instruction = data_line.get("a_to_b_instructions_eng", "")
        
        input_image_path = data_line["source"]
        input_image_raw = Image.open(input_image_path).convert("RGB")

        with torch.inference_mode():
            output = pipe(
                image=[input_image_raw],
                prompt=instruction,
                negative_prompt=" ",
                num_inference_steps=40,
                generator=torch.Generator(device=accelerator.device).manual_seed(
                    args.seed
                ),
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )

        image = output.images[0]
        image.save(data_line[args.save_key])

    print(
        f"[rank {accelerator.process_index}] "
        f"{len(data_list_process)} inferences finished"
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(
            f"[rank {accelerator.process_index}] "
            f"inference results saved at {args.save_path}"
        )
        with open(
            os.path.join(args.save_path, "result.jsonl"), "w", encoding="utf-8"
        ) as fout:
            for data_line in data_list:
                fout.write(json.dumps(data_line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--model-path",
        type=str,
        default="FireRedTeam/FireRed-Image-Edit-1.0",
    )
    arg_parser.add_argument(
        "--lang", 
        type=str, 
        choices=["cn", "eng"], 
        default="cn", 
        help="Language choice: cn or eng (default: cn)"
    )
    arg_parser.add_argument("--lora-name", type=str, default=None)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--jsonl-path", type=str, required=True)
    arg_parser.add_argument("--edit-task", type=str, default="all")
    arg_parser.add_argument("--save-key", type=str, default="result")
    arg_parser.add_argument("--seed", type=int, default=43)
    arg_parser.add_argument("--multi-folder", action="store_true")

    args = arg_parser.parse_args()

    main(args)