import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    """
    Main training function to fine-tune Qwen2.5-0.5B using QLoRA.
    Includes debug prints for step-by-step monitoring on Windows.
    """
    print(">>> DEBUG: Entering train() function...")
    
    # Clear CUDA memory before starting to prevent fragmentation
    if torch.cuda.is_available():
        print(">>> DEBUG: Clearing CUDA cache...")
        torch.cuda.empty_cache()
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    data_file = "arxiv_cs_2000.jsonl"
    output_dir = "./qwen-resercher-checkpoints"

    # 1. Load Tokenizer
    print(f">>> DEBUG: Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. BitsAndBytes Configuration (QLoRA)
    print(">>> DEBUG: Configuring BitsAndBytes for 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Base Model
    print(f">>> DEBUG: Loading base model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(">>> DEBUG: Preparing model for kbit training...")
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    print(">>> DEBUG: Setting up LoRA configuration (Rank=16)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Load Dataset
    print(f">>> DEBUG: Loading dataset from {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")

    # 6. Training Arguments
    print(">>> DEBUG: Defining training arguments...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,
        save_steps=100,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        packing=False
    )

    # 7. Initialize Trainer
    print(">>> DEBUG: Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # 8. Start Training
    print(">>> DEBUG: Starting training loop...")
    trainer.train()

    # 9. Save the Adapter
    print(">>> DEBUG: Saving the fine-tuned adapter to ./qwen-resercher...")
    trainer.model.save_pretrained("qwen-resercher")
    tokenizer.save_pretrained("qwen-resercher")
    print(">>> DEBUG: Training complete.")

if __name__ == "__main__":
    train()
