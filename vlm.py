from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

class VLM:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # We have to do a lot of work for this one because for now it's not predicting
    def predict_move(self, image_path, available_moves):
        moves_str = ", ".join([f"{m[0]} {m[1]} {m[2]}" for m in available_moves])
        question = (
            "You are playing Dots and Boxes. Choose one move "
            "from this list:\n" + moves_str +
            "\nRespond in the format: <type> <i> <j>\n"
            "Example: h 2 1"
        )

        img = Image.open(image_path)

        prompt = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}
        ]}]

        inputs = self.processor.apply_chat_template(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=30)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        for m in available_moves:
            formatted = f"{m[0]} {m[1]} {m[2]}"
            if formatted in answer:
                return m

        return available_moves[0]
