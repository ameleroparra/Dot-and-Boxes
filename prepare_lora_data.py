import json
import os

RAW_DIR = "data/raw"
OUT_FILE = "data/lora_train.jsonl"

os.makedirs("data", exist_ok=True)

samples = []

for game in sorted(os.listdir(RAW_DIR)):
    game_dir = os.path.join(RAW_DIR, game)
    if not os.path.isdir(game_dir):
        continue

    for file in sorted(os.listdir(game_dir)):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(game_dir, file), "r") as f:
            state = json.load(f)

        # Skip turns with no move
        if state["move_taken"] is None:
            continue

        img_path = state["screenshot"]
        move = state["move_taken"]
        available = state["move_info"]["available_moves"]

        moves_str = ", ".join([f"{m[0]} {m[1]} {m[2]}" for m in available])

        prompt = (
            "You are playing Dots and Boxes.\n"
            "Choose one move from this list:\n"
            f"{moves_str}\n"
            "Respond in the format: <type> <i> <j>"
        )

        answer = f"{move[0]} {move[1]} {move[2]}"

        samples.append({
            "image": img_path,
            "prompt": prompt,
            "answer": answer
        })

with open(OUT_FILE, "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print(f"Saved {len(samples)} samples to {OUT_FILE}")
