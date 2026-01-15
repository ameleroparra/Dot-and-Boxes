import os
import json
from typing import List, Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def collect_all_turns() -> List[Dict]:
    """Collect training examples where boxes are completed (using previous turn's state as input)."""
    all_turns = []
    
    game_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("game_") and os.path.isdir(os.path.join(DATA_DIR, d))])
    
    print(f"Found {len(game_dirs)} game directories")
    
    for game_dir in game_dirs:
        game_path = os.path.join(DATA_DIR, game_dir)
        turn_files = sorted([f for f in os.listdir(game_path) if f.startswith("turn_") and f.endswith(".json")])
        
        for i, turn_file in enumerate(turn_files):
            turn_path = os.path.join(game_path, turn_file)
            
            try:
                with open(turn_path, 'r') as f:
                    turn_data = json.load(f)
                
                # Check if this turn completed a box
                completed_box = turn_data.get('strategy_info', {}).get('completed_boxes_this_turn', False)
                
                if completed_box and i > 0:
                    # Get the PREVIOUS turn (i-1)
                    prev_turn_file = turn_files[i - 1]
                    prev_turn_path = os.path.join(game_path, prev_turn_file)
                    
                    with open(prev_turn_path, 'r') as f:
                        prev_turn_data = json.load(f)
                    
                    # Use previous turn's state as input, current turn's move as target
                    screenshot_path = os.path.join(SCRIPT_DIR, prev_turn_data.get('screenshot', ''))
                    
                    all_turns.append({
                        'json_path': prev_turn_path,
                        'screenshot_path': screenshot_path,
                        'game_id': prev_turn_data['game_id'],
                        'turn': prev_turn_data['turn'],
                        'current_player': prev_turn_data['current_player'],
                        'move_taken': turn_data['move_taken'],  # Move from CURRENT turn (completes box)
                        'available_moves': prev_turn_data['move_info']['available_moves'],
                        'scores': prev_turn_data['scores']
                    })
                    
            except Exception as e:
                print(f"Error processing {turn_path}: {e}")
    
    print(f"Collected {len(all_turns)} box-completing turns")
    return all_turns

def split_dataset(turns: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    total = len(turns)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    
    train_set = turns[:train_size]
    val_set = turns[train_size:train_size + val_size]
    test_set = turns[train_size + val_size:]
    
    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split_data, split_name in [(train_set, "train"), (val_set, "val"), (test_set, "test")]:
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} examples to {output_path}")
    
    return train_set, val_set, test_set

def print_statistics(train_set: List[Dict], val_set: List[Dict], test_set: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total = len(train_set) + len(val_set) + len(test_set)
    
    print(f"Total samples: {total} (box-completing moves only)")
    print(f"  Train:      {len(train_set)} ({len(train_set)/total*100:.1f}%)")
    print(f"  Validation: {len(val_set)} ({len(val_set)/total*100:.1f}%)")
    print(f"  Test:       {len(test_set)} ({len(test_set)/total*100:.1f}%)")


def main():
    print("Collecting all turns from games...")
    all_turns = collect_all_turns()
    
    if len(all_turns) == 0:
        print("No valid turns found. Exiting.")
        return
    
    print("\nSplitting and saving dataset...")
    train_set, val_set, test_set = split_dataset(all_turns)
    
    print_statistics(train_set, val_set, test_set)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
