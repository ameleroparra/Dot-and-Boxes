"""
Quick test to compare base VLM vs fine-tuned VLM on sample moves.
Run this after training to verify the model learned something useful.
"""

import json
import os
from vlm import VLM
from vlm_finetuned import VLMFineTuned

def test_on_sample_data(num_samples=10):
    """Test both models on the same data and compare."""
    
    print("Loading models...")
    base_model = VLM()
    finetuned_model = VLMFineTuned()
    print("✓ Models loaded\n")
    
    # Load some test samples
    data_file = "data/lora_train.jsonl"
    samples = []
    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    
    print(f"Testing on {len(samples)} samples...\n")
    print("="*70)
    
    base_correct = 0
    finetuned_correct = 0
    
    for i, sample in enumerate(samples):
        img_path = sample["image"]
        correct_answer = sample["answer"]
        
        # Extract available moves from prompt
        # Parse from: "Choose one move from this list:\nh 0 0, v 1 1, ..."
        prompt_lines = sample["prompt"].split("\n")
        moves_str = prompt_lines[1]  # Second line has the moves
        available_moves = []
        for move_str in moves_str.split(", "):
            parts = move_str.strip().split()
            if len(parts) == 3:
                available_moves.append((parts[0], int(parts[1]), int(parts[2])))
        
        # Get predictions
        base_pred = base_model.predict_move(img_path, available_moves)
        finetuned_pred = finetuned_model.predict_move(img_path, available_moves)
        
        base_pred_str = f"{base_pred[0]} {base_pred[1]} {base_pred[2]}"
        finetuned_pred_str = f"{finetuned_pred[0]} {finetuned_pred[1]} {finetuned_pred[2]}"
        
        base_is_correct = (base_pred_str == correct_answer)
        finetuned_is_correct = (finetuned_pred_str == correct_answer)
        
        if base_is_correct:
            base_correct += 1
        if finetuned_is_correct:
            finetuned_correct += 1
        
        print(f"Sample {i+1}:")
        print(f"  Correct:    {correct_answer}")
        print(f"  Base:       {base_pred_str} {'✓' if base_is_correct else '✗'}")
        print(f"  Fine-tuned: {finetuned_pred_str} {'✓' if finetuned_is_correct else '✗'}")
        print()
    
    print("="*70)
    print(f"RESULTS:")
    print(f"  Base Model:       {base_correct}/{len(samples)} correct ({base_correct/len(samples)*100:.1f}%)")
    print(f"  Fine-tuned Model: {finetuned_correct}/{len(samples)} correct ({finetuned_correct/len(samples)*100:.1f}%)")
    print()
    
    if finetuned_correct > base_correct:
        improvement = (finetuned_correct - base_correct) / len(samples) * 100
        print(f"✓ Fine-tuning IMPROVED performance by {improvement:.1f}%")
    elif finetuned_correct < base_correct:
        decline = (base_correct - finetuned_correct) / len(samples) * 100
        print(f"✗ Fine-tuning DECLINED performance by {decline:.1f}%")
        print("  → Need more/better training data or different hyperparameters")
    else:
        print("= No change in performance")

if __name__ == "__main__":
    test_on_sample_data(num_samples=20)
