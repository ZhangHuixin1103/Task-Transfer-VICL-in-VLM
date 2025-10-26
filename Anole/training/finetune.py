import torch
import jsonlines
from torch.utils.data import ConcatDataset
from transformers import (
    TrainingArguments, 
    ChameleonProcessor, 
    ChameleonForConditionalGeneration
)
from torch.nn.utils.rnn import pad_sequence
from perceptual import PerceptualLoss
from trainer import AnoleTrainer
from custom_datasets import InterleavedDataset, collate_fn
import argparse


parser = argparse.ArgumentParser(description="Training Chameleon")
parser.add_argument('--initial_model', '-i', required=True, help='Path to initial model')
parser.add_argument('--trained_model', '-o', required=True, help='Path to trained model')
args = parser.parse_args()


ANOLE_INITIAL_MODEL = args.initial_model
ANOLE_TRAINED_MODEL = args.trained_model
processor = ChameleonProcessor.from_pretrained(ANOLE_INITIAL_MODEL)

# Initialize the dataset
img_critique_data=InterleavedDataset("./dataset_example.jsonl")
dataset=img_critique_data
print(f"Total dataset length: {len(dataset)}")

# discriminator
loss_net: PerceptualLoss = PerceptualLoss()

# Define training arguments
training_args = TrainingArguments(
    output_dir=ANOLE_TRAINED_MODEL,
    learning_rate=1e-5,
    num_train_epochs=20,
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=4,
    save_steps=100000,
    bf16=True,
    #report_to="tensorboard",
    logging_steps=1,
    deepspeed="ds_config.json",
    #save_strategy="epoch"
    # max_grad_norm=10
)

print("training_args is bf16?", training_args.bf16)

# Initialize the model
model = ChameleonForConditionalGeneration.from_pretrained(
    ANOLE_INITIAL_MODEL,
    torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    # attn_implementation="flash_attention_2",
)

print("attention implementation", model.config._attn_implementation)

# Initialize the Trainer with custom collate_fn

trainer = AnoleTrainer(
    loss_net=loss_net,
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

# Train the model
trainer.train()

# Save the final model checkpoint
trainer.save_model()

# Save the model
processor.save_pretrained(ANOLE_TRAINED_MODEL)