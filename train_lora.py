import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List
from dataclasses import dataclass

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

# LoRA Configuration
LORA_R = 16  # Rank of the LoRA matrices
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention modules

# Training Configuration
LEARNING_RATE = 2e-4
BATCH_SIZE = 4  # Adjust based on your GPU memory
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 100
EVAL_STEPS = 100


class DotsAndBoxesDataset(Dataset):
    """Dataset for Dots and Boxes move prediction."""
    
    def __init__(self, data_file: str, processor, max_length: int = 1024):

        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.processor = processor 
        self.max_length = max_length
        print(f"Loaded {len(self.data)} examples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training example."""
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['screenshot_path']).convert('RGB')
        
        # Format available moves
        available_moves = item['available_moves']
        move_list = [f"{m[0]} {m[1]} {m[2]}" for m in available_moves]
        available_moves_text = "\n=== AVAILABLE MOVES ===\n" + "\n".join(move_list) + "\n=== END OF AVAILABLE MOVES ==="
        
        # Determine player color
        player_color = "RED" if item['current_player'] == 0 else "BLUE"
        
        # Create instruction prompt
        instruction = f"""You are the {player_color} player in Dots and Boxes game.

IMPORTANT: You MUST choose EXACTLY one move from the available moves list below.
{available_moves_text}

Strategy:
1. Prioritize completing boxes for yourself.
2. Avoid placing the third edge when a square has already two edges.
3. If no boxes can be completed, minimize chances for opponent to complete boxes.
4. If multiple box chains exist, choose the move that allows you to claim the longest chain.

Reply with ONLY one move from the list above in format: h ROW COL or v ROW COL
Your move:"""
        
        # Format target move
        move = item['move_taken']
        target = f"{move[0]} {move[1]} {move[2]}"
        
        # Create conversation format expected by Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target}
                ]
            }
        ]
        
        # Process with the VLM processor
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Get processor inputs
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Prepare labels (same as input_ids for causal language modeling)
        labels = inputs['input_ids'].clone()
        
        # Mask the instruction part (only compute loss on the assistant's response)
        # This is a simplified approach - you may want to be more precise
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs.get('pixel_values', inputs.get('image_grid_thw')).squeeze(0) if 'pixel_values' in inputs or 'image_grid_thw' in inputs else None,
            'labels': labels.squeeze(0)
        }


def setup_lora_model(model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):
    """Load base model and apply LoRA configuration."""
    print(f"Loading base model: {model_name}")
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    print("="*60)
    print("Dots and Boxes VLM LoRA Fine-tuning")
    print("="*60)
    
    # Check if prepared data exists
    train_file = os.path.join(PREPARED_DATA_DIR, "train.json")
    val_file = os.path.join(PREPARED_DATA_DIR, "val.json")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("\nError: Prepared dataset not found!")
        print("Please run 'python prepare_dataset.py' first.")
        return
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        trust_remote_code=True
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DotsAndBoxesDataset(train_file, processor)
    val_dataset = DotsAndBoxesDataset(val_file, processor)
    
    # Setup model with LoRA
    print("\nSetting up LoRA model...")
    model = setup_lora_model()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        print(f"\nSaving final model to {final_model_path}")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Model saved to: {final_model_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
