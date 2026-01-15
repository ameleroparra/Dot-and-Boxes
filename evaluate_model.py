"""
Evaluate the fine-tuned LoRA model on the test set.

This script:
1. Loads the fine-tuned model
2. Runs inference on the test set
3. Calculates accuracy and other metrics
4. Provides detailed analysis of model performance
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")
MODEL_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "final_model")


def load_finetuned_model(model_path: str):
    """Load the fine-tuned LoRA model."""
    print(f"Loading fine-tuned model from {model_path}")
    
    # Load model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    return model, processor


def parse_move(text: str) -> Tuple[str, int, int]:
    """Parse move from model output."""
    # Look for pattern: h/v ROW COL
    match = re.search(r'\b([hv])\s+(\d+)\s+(\d+)', text.lower())
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None


def predict_move(model, processor, image_path: str, available_moves: List, current_player: int) -> str:
    """Predict a move for the given game state."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Format available moves
    move_list = [f"{m[0]} {m[1]} {m[2]}" for m in available_moves]
    available_moves_text = "\n=== AVAILABLE MOVES ===\n" + "\n".join(move_list) + "\n=== END OF AVAILABLE MOVES ==="
    
    # Determine player color
    player_color = "RED" if current_player == 0 else "BLUE"
    
    # Create prompt
    prompt = f"""You are the {player_color} player in Dots and Boxes game.

IMPORTANT: You MUST choose EXACTLY one move from the available moves list below.
{available_moves_text}

Strategy:
1. Prioritize completing boxes for yourself.
2. Avoid placing the third edge when a square has already two edges.
3. If no boxes can be completed, minimize chances for opponent to complete boxes.
4. If multiple box chains exist, choose the move that allows you to claim the longest chain.

Reply with ONLY one move from the list above in format: h ROW COL or v ROW COL
Your move:"""
    
    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process input
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    # Decode
    generated_text = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Extract just the assistant's response
    if "assistant" in generated_text.lower():
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text
    
    return response


def evaluate_test_set(model, processor, test_file: str):
    """Evaluate model on test set."""
    print(f"\nLoading test data from {test_file}")
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Evaluating on {len(test_data)} examples...\n")
    
    results = {
        'correct': 0,
        'incorrect': 0,
        'invalid': 0,  # Move not in available moves
        'parse_error': 0,  # Couldn't parse the output
        'total': len(test_data),
        'by_player': {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}},
        'errors': []
    }
    
    for i, item in enumerate(test_data):
        print(f"Processing {i+1}/{len(test_data)}...", end='\r')
        
        # Get prediction
        prediction_text = predict_move(
            model,
            processor,
            item['screenshot_path'],
            item['available_moves'],
            item['current_player']
        )
        
        # Parse prediction
        predicted_move = parse_move(prediction_text)
        true_move = tuple(item['move_taken'])
        
        player = item['current_player']
        results['by_player'][player]['total'] += 1
        
        if predicted_move is None:
            results['parse_error'] += 1
            results['errors'].append({
                'type': 'parse_error',
                'game_id': item['game_id'],
                'turn': item['turn'],
                'output': prediction_text,
                'expected': true_move
            })
        elif predicted_move not in [tuple(m) for m in item['available_moves']]:
            results['invalid'] += 1
            results['errors'].append({
                'type': 'invalid_move',
                'game_id': item['game_id'],
                'turn': item['turn'],
                'predicted': predicted_move,
                'expected': true_move
            })
        elif predicted_move == true_move:
            results['correct'] += 1
            results['by_player'][player]['correct'] += 1
        else:
            results['incorrect'] += 1
            results['errors'].append({
                'type': 'wrong_move',
                'game_id': item['game_id'],
                'turn': item['turn'],
                'predicted': predicted_move,
                'expected': true_move
            })
    
    print()  # New line after progress
    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    total = results['total']
    correct = results['correct']
    
    print(f"\nOverall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"\nBreakdown:")
    print(f"  Correct predictions:     {results['correct']} ({results['correct']/total*100:.2f}%)")
    print(f"  Incorrect predictions:   {results['incorrect']} ({results['incorrect']/total*100:.2f}%)")
    print(f"  Invalid moves:           {results['invalid']} ({results['invalid']/total*100:.2f}%)")
    print(f"  Parse errors:            {results['parse_error']} ({results['parse_error']/total*100:.2f}%)")
    
    print(f"\nBy Player:")
    for player, stats in results['by_player'].items():
        player_name = "RED" if player == 0 else "BLUE"
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {player_name} (player {player}): {stats['correct']}/{stats['total']} ({acc:.2f}%)")
    
    # Show some errors
    if results['errors']:
        print(f"\nSample Errors (showing first 5):")
        for error in results['errors'][:5]:
            print(f"\n  Type: {error['type']}")
            print(f"  Game {error['game_id']}, Turn {error['turn']}")
            if 'predicted' in error:
                print(f"  Predicted: {error['predicted']}")
            if 'output' in error:
                print(f"  Output: {error['output'][:100]}...")
            print(f"  Expected: {error['expected']}")
    
    print("\n" + "="*60)


def main():
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the model first using 'python train_lora.py'")
        return
    
    # Check if test data exists
    test_file = os.path.join(PREPARED_DATA_DIR, "test.json")
    if not os.path.exists(test_file):
        print(f"\nError: Test data not found at {test_file}")
        print("Please run 'python prepare_dataset.py' first")
        return
    
    # Load model
    model, processor = load_finetuned_model(MODEL_PATH)
    
    # Evaluate
    results = evaluate_test_set(model, processor, test_file)
    
    # Print results
    print_results(results)
    
    # Save results
    results_file = os.path.join(SCRIPT_DIR, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
