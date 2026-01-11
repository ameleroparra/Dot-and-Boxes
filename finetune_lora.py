import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image
import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset with train/val split
dataset = load_dataset("json", data_files="data/lora_train.jsonl")
# Split into train and validation
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["prompt"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["answer"]}],
        },
    ]

    # Build chat text
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # We have the assistant response
    )

    # Process multimodal inputs
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        truncation=True,
        padding=False,  
    )

    # CRITICAL FIX: Mask labels for non-assistant tokens
    # Only compute loss on assistant's response, not on user prompt or vision tokens
    input_ids = inputs["input_ids"][0]
    labels = input_ids.clone()
    
    # Find where assistant response starts
    # The assistant response comes after the user message
    # We need to mask everything before "assistant" token
    text_tokens = processor.tokenizer.encode(text, add_special_tokens=False)
    
    # Find assistant response start - look for the answer text in tokenization
    answer_text = example["answer"]
    answer_tokens = processor.tokenizer.encode(answer_text, add_special_tokens=False)
    
    # Find where answer starts in the full token sequence
    answer_start = None
    for i in range(len(input_ids) - len(answer_tokens) + 1):
        if input_ids[i:i+len(answer_tokens)].tolist() == answer_tokens:
            answer_start = i
            break
    
    # Mask everything before the answer
    if answer_start is not None:
        labels[:answer_start] = -100
    else:
        # Fallback: mask first 80% of tokens (vision + user prompt approximately)
        mask_until = int(len(labels) * 0.8)
        labels[:mask_until] = -100
    
    inputs["labels"] = labels.unsqueeze(0)

    # Remove batch dimension
    return {k: v.squeeze(0) for k, v in inputs.items()}

# Apply preprocessing to both splits
train_dataset = dataset["train"].map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

eval_dataset = dataset["test"].map(
    preprocess,
    remove_columns=dataset["test"].column_names
)

# Data collator for padding
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="lora-dots-boxes",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
    learning_rate=1e-4,  # REDUCED: Lower LR for stable fine-tuning
    num_train_epochs=5,  # More epochs with early stopping
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",  # NEW: Evaluate during training
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,  # Keep only 3 best checkpoints
    load_best_model_at_end=True,  # Load best model after training
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=50,  # Warmup for stable training
    weight_decay=0.01,  # Regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # NEW: Validation set
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("\n" + "="*50)
print("Training complete! Saving model...")
model.save_pretrained("lora-dots-boxes")
processor.save_pretrained("lora-dots-boxes")  # Save processor too
print("Model saved to lora-dots-boxes/")
print("="*50)
