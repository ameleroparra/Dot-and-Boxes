import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data/lora_train.jsonl")

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["prompt"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["answer"]}],
        },
    ]

    # 1️⃣ Build chat text (NO tokenization here)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
    )

    # 2️⃣ Proper multimodal processing
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        truncation=True,
    )

    # 3️⃣ Labels
    inputs["labels"] = inputs["input_ids"].clone()

    # 4️⃣ Remove batch dim (Trainer will re-batch)
    return {k: v.squeeze(0) for k, v in inputs.items()}



dataset = dataset["train"].map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir="lora-dots-boxes",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("lora-dots-boxes")
