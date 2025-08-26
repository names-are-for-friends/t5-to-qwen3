import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Configuration
DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset.txt"
ENHANCED_DATASET_PATH = "/mnt/f/q5_xxs_training_script/400K_dataset_enhanced.txt"
model_checkpoint = "gokaygokay/Flux-Prompt-Enhance"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_target_length = 460
prefix = "enhance prompt: "
batch_size = 64
PREFETCH_FACTOR = 16

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.to(device)
model.eval()

# Create a dataset class
class PromptDataset(Dataset):
    def __init__(self, lines, prefix):
        self.lines = lines
        self.prefix = prefix

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.prefix + self.lines[idx]

# Read the original dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

dataset = PromptDataset(lines, prefix)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=min(4, os.cpu_count()//2) if torch.cuda.is_available() else 0,
    persistent_workers=False,
    prefetch_factor=PREFETCH_FACTOR,
)

# Open the output file for writing enhanced prompts
with open(ENHANCED_DATASET_PATH, "w", encoding="utf-8") as out_f:
    for batch in tqdm(dataloader, desc="Enhancing prompts"):
        inputs = batch  # list of strings with prefix
        # Tokenize the batch
        tokens = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Generate enhanced prompts
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                repetition_penalty=1.2,
                do_sample=True,
                temperature=0.7,
            )

        # Decode the outputs
        enhanced_prompts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Write each enhanced prompt to the file
        for prompt in enhanced_prompts:
            out_f.write(prompt + "\n")

        # Print for logging (first item in the batch)
        print(f"Input: {inputs[0]}")
        print(f"Output: {enhanced_prompts[0]}")
        print("-----")

print(f"Enhanced prompts saved to {ENHANCED_DATASET_PATH}")
