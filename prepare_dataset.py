import json
import random
from pathlib import Path

base_path = Path("data/raw")
prepared_path = base_path.parent / "prepared"
prepared_path.mkdir(parents=True, exist_ok=True)

dataset = []

# All the logic where we select only moves when a box was created
for game_dir in sorted(base_path.glob("game_*")):
    
    turn_files = sorted(list(game_dir.glob("turn_*.json")))

    for i in range(1, len(turn_files)):
        current_file = turn_files[i]
        previous_file = turn_files[i-1]

        with open(current_file, 'r') as f_curr:

            current_data = json.load(f_curr)

            if current_data.get("strategy_info", {}).get("completed_boxes_this_turn") is True:
                
                with open(previous_file, 'r') as f_prev:
                    previous_data = json.load(f_prev)
                
                img_path = previous_data.get("screenshot")
                moves = previous_data.get("move_info", {}).get("available_moves")
                answer = current_data.get("move_taken")
                player = current_data.get("current_player")

                player = "Red" if player == 0 else "Blue"

                conversation_entry = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": f"You are playing Dots and Boxes, you are player {player}. The image you are seing is what the previous player did. The other player placed a third edge and now there is an option for you to complete a box and win a point. To select a move you will have first to decide if you want to place an horizontal (h) or a vertical (v) line between the grey points. Then you will have to specify in which position you want to place that line with the next format: ['h', x, y] or ['v', x, y] where x is the row and y is the column. For horizontal lines, rows go from top(0) to bottom(4) and columns from left(0) to right(3). For vertical lines, rows go from top(0) to bottom(3) and columns from left(0) to right(4). Your available moves are: {moves}. choose the move that will complete the box."}
                            ]
                    
                        },
                        {
                            "role": "assistant",
                            "content":[
                                {"type": "text", "text": f"{answer}"}
                            ]
                        }
                    ]
                }
                dataset.append(conversation_entry)

# We shuffle with a defined seed
random.seed(42)
random.shuffle(dataset)

# We split into train (80%), eval (10%), test (10%)
total_len = len(dataset)
train_end = int(total_len * 0.8)
eval_end = int(total_len * 0.9)

train_data = dataset[:train_end]
eval_data = dataset[train_end:eval_end]
test_data = dataset[eval_end:]

splits = {
    "dataset_train.json": train_data,
    "dataset_eval.json": eval_data,
    "dataset_test.json": test_data
}

# Here we save the splits
for filename, data_split in splits.items():
    save_path = prepared_path / filename
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_split, f, indent=4)


# some info
print(f"Dataset processing complete!")
print(f"Total instances: {total_len}")
print(f"Saved: {len(train_data)} to train, {len(eval_data)} to eval, {len(test_data)} to test.")