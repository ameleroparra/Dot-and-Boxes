from transformers import AutoModelForImageTextToText, AutoProcessor
import json
import re
import os

class VLM:
    def __init__(self):
        print("Loading Qwen3-VL-4B...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        print("VLM ready!")
    
    def get_move(self, screenshot_path, json_path=None):
        
        # Extract available moves from JSON
        available_moves_text = ""
        moves = []
        if json_path and os.path.exists(json_path):
            print(f"DEBUG: Loading JSON from {json_path}")
            with open(json_path) as f:
                state = json.load(f)
                
                # Get available moves only
                if 'move_info' in state and 'available_moves' in state['move_info']:
                    moves = state['move_info']['available_moves']
                    print(f"DEBUG: Found {len(moves)} available moves in JSON")
                    print(f"DEBUG: First 10 moves: {moves[:10]}")
                    move_list = [f"{m[0]} {m[1]} {m[2]}" for m in moves]
                    available_moves_text = f"\n\n=== AVAILABLE MOVES (You MUST choose from this list) ===\n" + "\n".join(move_list) + "\n=== END OF AVAILABLE MOVES ==="
                else:
                    print("DEBUG: No available moves found in JSON!")
        else:
            print(f"DEBUG: JSON file not found: {json_path}")
        
        prompt_text = f"""You are the BLUE player in Dots and Boxes game. You are seeing what the Red player did in the turn before you. 

IMPORTANT: You MUST choose EXACTLY one move from the available moves list below. Do NOT make up a move. Do NOT choose a move that is not in the list.
{available_moves_text}

Strategy:
1. Prioritize completing boxes for yourself (BLUE).
2. Avoid placing the third edge when a square has already two edges, because it allows the opponent (RED) to complete the box in their next turn.
3. If no boxes can be completed, choose moves that minimize the chances for the opponent to complete boxes in their next turn.
4. If more than one chain of boxes exist, choose the move that allows you to claim the longest chain of boxes.

Reply with ONLY one move from the list above in format: h ROW COL or v ROW COL
Your move:"""
        
        print("=" * 80)
        print("PROMPT SENT TO VLM:")
        print(prompt_text)
        print("=" * 80)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot_path},
                {"type": "text", "text": prompt_text},
            ],
        }]
        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print("=" * 80)
        print("RAW VLM OUTPUT:")
        print(response)
        print("=" * 80)
        
        # Parse response
        match = re.search(r'([hv])\s+(\d+)\s+(\d+)', response.lower())
        if match:
            line_type = match.group(1)
            row = int(match.group(2))
            col = int(match.group(3))
            print(f"PARSED MOVE: type={line_type}, row={row}, col={col}")
            return (line_type, row, col)
        
        print("Failed to parse VLM response - no valid move pattern found")
        return None
