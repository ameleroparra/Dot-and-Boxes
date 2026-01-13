import json
import os

RAW_DIR = "data/raw"
OUT_FILE = "data/lora_train.jsonl"

os.makedirs("data", exist_ok=True)

def is_good_move(state):
    
    # Skip if no move was taken
    if state["move_taken"] is None:
        return False
    
    # include mid-to-late game moves
    game_phase = state["strategy_info"].get("game_phase", "early")
    if game_phase == "early":
        return False
    
    # Inlclude moves where player completed boxes
    if state["strategy_info"].get("completed_boxes_this_turn", 0) > 0:
        return True
    
    # Include moves from late game
    if game_phase == "late":
        return True
    
    # For mid-game, include moves where there are potential boxes (strategic decisions)
    potential = state["strategy_info"].get("potential_boxes", {})
    if potential.get("three_edges", 0) > 0 or potential.get("two_edges", 0) > 1:
        return True
    
    return False  # Skip otherwise

samples = []
skipped = 0

for game in sorted(os.listdir(RAW_DIR)):
    game_dir = os.path.join(RAW_DIR, game)
    if not os.path.isdir(game_dir):
        continue

    for file in sorted(os.listdir(game_dir)):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(game_dir, file), "r") as f:
            state = json.load(f)

        if not is_good_move(state):
            skipped += 1
            continue

        img_path = state["screenshot"]
        move = state["move_taken"]
        available = state["move_info"]["available_moves"] + [move]

        moves_str = ", ".join([f"{m[0]} {m[1]} {m[2]}" for m in available])

        prompt = (
            "You are playing Dots and Boxes and must prioritize chain control by using the 'Double-Cross' strategy, sacrificing the end of a current chain to force your opponent to open the next segment so you can capture the final, longest chain; Avoid when possible placing the third edge on a square that has already 2 edges."
            "Select the best move from this list:\n"
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

print(f"✓ Saved {len(samples)} high-quality samples to {OUT_FILE}")
print(f"  Skipped {skipped} low-quality samples (early game random moves)")
print(f"  Training data quality: {len(samples)/(len(samples)+skipped)*100:.1f}%")
