from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch
from PIL import Image

class VLMFineTuned:
    """Fine-tuned VLM model for Dots and Boxes game."""
    
    def __init__(self):
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        lora_path = "lora-dots-boxes"
        
        print("Loading fine-tuned VLM model...")
        
        # Load processor from base model (not from LoRA path)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        
        # Load base model first WITHOUT device_map to avoid structure issues
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load fine-tuned LoRA weights on top of base model
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        
        # Move to device after LoRA is loaded
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        print("✓ Fine-tuned VLM model loaded successfully!")

    def predict_move(self, image_path, available_moves):
        """Predict the best move given a board screenshot and available moves."""
        
        # Format available moves for prompt
        moves_str = ", ".join([f"{m[0]} {m[1]} {m[2]}" for m in available_moves])
        question = (
            "You are playing Dots and Boxes and must prioritize chain control by using the 'Double-Cross' strategy, sacrificing the end of a current chain to force your opponent to open the next segment so you can capture the final, longest chain; Avoid when possible placing the third edge on a square that has already 2 edges."
            "Select the best move from this list:\n"
            f"{moves_str}\n"
            "Respond in the format: <type> <i> <j>"
        )

        # Load and prepare image
        img = Image.open(image_path)

        # Create prompt
        prompt = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}
        ]}]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate prediction
        output = self.model.generate(**inputs, max_new_tokens=30)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        # Parse the answer to find a valid move
        for m in available_moves:
            formatted = f"{m[0]} {m[1]} {m[2]}"
            if formatted in answer:
                return m

        # Fallback to first available move if parsing fails
        return available_moves[0]
